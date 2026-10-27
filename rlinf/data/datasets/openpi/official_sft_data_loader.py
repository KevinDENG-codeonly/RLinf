# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Iterator

import numpy as np
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    norm_stats_path_from_data_kwargs,
    select_openpi_norm_stats,
)


class _OpenPIWorkerDataset(torch.utils.data.Dataset):
    """Include worker-side random transforms in consumed-data snapshots."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def state_dict(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

    def load_state_dict(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])


class StatefulOpenPIDataLoader(StatefulDataLoader):
    """Finite OpenPI epochs using the existing SFT checkpoint contract."""

    def __init__(
        self, dataset: Any, data_config: Any, signature: dict[str, Any], **kwargs: Any
    ) -> None:
        super().__init__(dataset, **kwargs)
        self._config = data_config
        self._signature = signature
        self._restored_generator = None
        if not len(self):
            raise ValueError("Expected at least one complete OpenPI batch per rank.")

    def data_config(self) -> Any:
        """Return the OpenPI transforms and normalization configuration."""
        return self._config

    def __iter__(self) -> Iterator[tuple[Any, torch.Tensor]]:
        from openpi.models.model import Observation
        from torch.utils._pytree import tree_map

        iterator = super().__iter__()
        if self._restored_generator is not None:
            self.generator.set_state(self._restored_generator)
            self._restored_generator = None
        for batch in iterator:
            batch = tree_map(torch.as_tensor, batch)
            yield Observation.from_dict(batch), batch["actions"]

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the sampler, prefetched data, and worker random states."""
        return {
            "signature": self._signature,
            "epoch": self.sampler.epoch,
            "generator": self.generator.get_state(),
            "loader": super().state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore only with unchanged data, preprocessing, and rank layout."""
        if state["signature"] != self._signature:
            raise ValueError(
                "OpenPI loader restore requires unchanged dataset, transforms, "
                "normalization, rank layout, batch size, workers, and seed."
            )
        self.sampler.set_epoch(state["epoch"])
        super().load_state_dict(state["loader"])
        self.generator.set_state(state["generator"])
        self._restored_generator = state["generator"]


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    if not norm_stats_path_from_data_kwargs(data_kwargs):
        select_openpi_norm_stats(None, norm_stats_path=None)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=data_kwargs,
    )
    config = dataclasses.replace(
        config,
        num_workers=int(
            OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
        ),
        seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
    )
    _validate_openpi_model_shape(model_cfg, config)

    if OmegaConf.select(cfg, "data.stateful_loader", default=False):
        import hashlib
        from pathlib import Path

        for name in ("max_token_len", "discrete_state_input"):
            value = OmegaConf.select(model_cfg.openpi, name, default=None)
            expected = getattr(config.model, name)
            if value is not None and value != expected:
                raise ValueError(
                    f"Expected openpi.{name}={expected} from "
                    f"{config.name}, got {value}."
                )
        data_config = config.data.create(config.assets_dirs, config.model)
        if data_config.rlds_data_dir is not None:
            raise ValueError(
                "Stateful OpenPI loading requires a map-style LeRobot dataset."
            )
        dataset = openpi_data_loader.create_torch_dataset(
            data_config, config.model.action_horizon, config.model
        )
        dataset = openpi_data_loader.transform_dataset(dataset, data_config)
        norm_path = norm_stats_path_from_data_kwargs(data_kwargs)
        if not norm_path:
            raise ValueError(
                "Stateful OpenPI loading requires explicit norm_stats_path."
            )
        norm_path = Path(norm_path)
        if norm_path.is_dir():
            norm_path = norm_path / "norm_stats.json"
        signature = {
            "repo_id": repo_id,
            "length": len(dataset),
            "world_size": world_size,
            "rank": rank,
            "batch_size": batch_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "shuffle": not eval_dataset,
            "config_name": model_cfg.openpi.config_name,
            "model": dataclasses.asdict(config.model),
            "data_kwargs": data_kwargs,
            "norm_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        }
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=not eval_dataset,
            seed=config.seed,
            drop_last=True,
        )
        data_loader = StatefulOpenPIDataLoader(
            _OpenPIWorkerDataset(dataset),
            data_config,
            signature,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=openpi_data_loader._collate_fn,
            worker_init_fn=openpi_data_loader._worker_init_fn,
            generator=torch.Generator().manual_seed(config.seed),
            multiprocessing_context="spawn" if config.num_workers else None,
            drop_last=True,
            snapshot_every_n_steps=1,
        )
    else:
        data_loader = openpi_data_loader.create_data_loader(
            config, framework="pytorch", shuffle=not eval_dataset
        )
    return data_loader, data_loader.data_config()


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    if isinstance(data_loader, StatefulOpenPIDataLoader):
        return len(data_loader)
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    return len(torch_loader)


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return getattr(data_loader, "_data_loader", None) is not None


def _validate_openpi_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config.

    The official loader sizes the SFT action window from
    ``TrainConfig.model.action_horizon``, not ``num_action_chunks`` (that field
    is the env-executed chunk). ``get_model`` already builds the network from
    the same TrainConfig unless YAML overrides ``openpi.action_horizon``.
    """
    yaml_horizon = OmegaConf.select(model_cfg, "openpi.action_horizon", default=None)
    official_horizon = int(openpi_config.model.action_horizon)
    if yaml_horizon is not None and int(yaml_horizon) != official_horizon:
        raise ValueError(
            "openpi SFT data uses TrainConfig.model.action_horizon="
            f"{official_horizon} from {model_cfg.openpi.config_name}; "
            f"openpi.action_horizon={int(yaml_horizon)} would build a different "
            "network. Unset it, or change the TrainConfig."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )
