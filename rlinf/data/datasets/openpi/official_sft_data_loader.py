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
from rlinf.models.embodiment.openpi.hierarchical import HierarchicalConfig
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    norm_stats_path_from_data_kwargs,
    select_openpi_norm_stats,
)


class _OpenPIWorkerDataset(torch.utils.data.Dataset):
    """Include worker-side random transforms in consumed-data snapshots."""

    def __init__(self, dataset: Any, malloc_trim_interval: int = 0) -> None:
        if (
            isinstance(malloc_trim_interval, bool)
            or not isinstance(malloc_trim_interval, int)
            or malloc_trim_interval < 0
        ):
            raise ValueError(
                "Expected a nonnegative integer worker_malloc_trim_interval, "
                f"got {malloc_trim_interval!r}."
            )
        self.dataset = dataset
        self._malloc_trim_interval = malloc_trim_interval
        self._reads = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        value = self.dataset[index]
        if (
            self._malloc_trim_interval
            and torch.utils.data.get_worker_info() is not None
        ):
            self._reads += 1
            if self._reads % self._malloc_trim_interval == 0:
                import ctypes

                try:
                    trim = ctypes.CDLL(None).malloc_trim
                except AttributeError as exc:
                    raise RuntimeError(
                        "worker_malloc_trim_interval requires glibc malloc_trim; "
                        "set it to 0 on unsupported allocators."
                    ) from exc
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)
        return value

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
        self._hierarchical = signature.get("hierarchical", {}).get("enabled", False)
        self._objective_offset = 0
        if not len(self):
            raise ValueError("Expected at least one complete OpenPI batch per rank.")

    def data_config(self) -> Any:
        """Return the OpenPI transforms and normalization configuration."""
        return self._config

    def __iter__(self) -> Iterator[tuple[Any, torch.Tensor] | dict[str, Any]]:
        from openpi.models.model import Observation
        from torch.utils._pytree import tree_map

        iterator = super().__iter__()
        if self._restored_generator is not None:
            self.generator.set_state(self._restored_generator)
            self._restored_generator = None
        for batch in iterator:
            batch = tree_map(torch.as_tensor, batch)
            observation = Observation.from_dict(batch)
            if self._hierarchical:
                objective = "subtask" if self._objective_offset % 2 == 0 else "flow"
                self._objective_offset += 1
                yield {
                    "observation": observation,
                    "actions": batch["actions"],
                    "objective": objective,
                    **{
                        key: value
                        for key, value in batch.items()
                        if key.startswith("high_level_")
                    },
                }
            else:
                yield observation, batch["actions"]

    def state_dict(self) -> dict[str, Any]:
        """Snapshot the sampler, prefetched data, and worker random states."""
        state = {
            "signature": self._signature,
            "epoch": self.sampler.epoch,
            "generator": self.generator.get_state(),
            "loader": super().state_dict(),
        }
        if self._hierarchical:
            state["objective_offset"] = self._objective_offset
        return state

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
        if self._hierarchical:
            self._objective_offset = state["objective_offset"]


def _validate_hierarchical_normalization(data_config: Any, zero_base: bool) -> None:
    """Require the stationary-base transform when its action stats encode zero."""
    if zero_base:
        return
    from rlinf.models.embodiment.openpi.policies.spiritai_policy import ACTION_DIM

    stats = (data_config.norm_stats or {}).get("actions")
    if stats is None or stats.q01 is None or stats.q99 is None:
        return
    lower = np.asarray(stats.q01)[ACTION_DIM - 3 : ACTION_DIM]
    upper = np.asarray(stats.q99)[ACTION_DIM - 3 : ACTION_DIM]
    if lower.shape == (3,) and np.all(lower == 0) and np.all(upper == 0):
        raise ValueError(
            "FoldBox action stats encode zero base commands. Set "
            "openpi_data.zero_base_action_targets=true to match them, or supply "
            "statistics for the intended nonzero base-action targets."
        )


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

    stateful_loader = OmegaConf.select(cfg, "data.stateful_loader", default=False)
    hierarchy = HierarchicalConfig.from_mapping(model_cfg.openpi.get("hierarchical"))
    if hierarchy.enabled:
        accumulation = cfg.actor.global_batch_size // (batch_size * world_size)
        if (
            not stateful_loader
            or cfg.actor.global_batch_size % (batch_size * world_size)
            or accumulation < 2
            or accumulation % 2
        ):
            raise ValueError(
                "Hierarchical SFT requires stateful_loader=true and an even "
                "gradient accumulation count >= 2 for one HL/LL pair per update."
            )
        if model_cfg.openpi.config_name != "pi05_spiritai_cartesian":
            raise ValueError(
                "Hierarchical SFT currently requires pi05_spiritai_cartesian."
            )
    trim_interval = OmegaConf.select(cfg, "data.worker_malloc_trim_interval", default=0)
    if trim_interval != 0 and not stateful_loader:
        raise ValueError(
            "worker_malloc_trim_interval requires data.stateful_loader=true."
        )
    if stateful_loader:
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
        if hierarchy.enabled:
            _validate_hierarchical_normalization(
                data_config,
                bool((data_kwargs or {}).get("zero_base_action_targets", False)),
            )
            from openpi.transforms import compose

            from rlinf.models.embodiment.openpi.hierarchical import (
                HierarchicalInputs,
                SubtaskTokenizer,
                TokenizeHighLevel,
            )

            data_config = dataclasses.replace(
                data_config,
                data_transforms=dataclasses.replace(
                    data_config.data_transforms,
                    inputs=[
                        HierarchicalInputs(compose(data_config.data_transforms.inputs))
                    ],
                ),
                model_transforms=dataclasses.replace(
                    data_config.model_transforms,
                    inputs=[
                        TokenizeHighLevel(
                            SubtaskTokenizer(
                                config.model.max_token_len, hierarchy.max_subtask_tokens
                            ),
                            config.model.discrete_state_input,
                        ),
                        *data_config.model_transforms.inputs,
                    ],
                ),
            )
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
        if hierarchy.enabled:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

            root = LeRobotDatasetMetadata(repo_id).root
            fingerprint = hashlib.sha256()
            paths = [
                root / "meta/info.json",
                root / "meta/tasks.jsonl",
                root / "meta/episodes.jsonl",
            ]
            paths.extend(sorted((root / "data").rglob("*.parquet")))
            for path in paths:
                fingerprint.update(str(path.relative_to(root)).encode())
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        fingerprint.update(block)
            signature["hierarchical"] = dataclasses.asdict(hierarchy)
            signature["hierarchical_format_version"] = 1
            signature["data_sha256"] = fingerprint.hexdigest()
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=not eval_dataset,
            seed=config.seed,
            drop_last=True,
        )
        data_loader = StatefulOpenPIDataLoader(
            _OpenPIWorkerDataset(dataset, malloc_trim_interval=trim_interval),
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
