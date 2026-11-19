# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare deployment provenance from local OpenPI checkpoint and W&B records."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi.checkpoint import (
    DEPLOYMENT_FIELDS,
    _normalize_state_dict,
    file_sha256,
    resolve_full_weights,
)
from rlinf.utils.logging import get_logger


def prepare_deployment(
    checkpoint: str | Path,
    output: str | Path,
    *,
    policy_kind: str,
    training_config: str | Path,
    launch_metadata: str | Path,
    norm_stats_path: str | Path | None = None,
) -> Path:
    """Write a new metadata bundle referencing immutable, unmerged-free weights.

    Read only trusted local checkpoint files. Full Flow requires its saved
    DataLoader signature; explicit Subtask requires merged training metadata.
    Existing bundles are never overwritten and weights are never rewritten.
    """
    import torch

    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    if policy_kind not in ("flow_only", "explicit_subtask"):
        raise ValueError(
            f"Expected flow_only or explicit_subtask, got {policy_kind!r}."
        )
    destination = Path(output).expanduser()
    if destination.exists():
        raise FileExistsError(
            f"Expected a new deployment directory, got {destination}."
        )
    weights = resolve_full_weights(checkpoint)
    if weights is None:
        raise FileNotFoundError(f"Expected native full_weights.pt under {checkpoint}.")
    weights = weights.resolve()
    state = _normalize_state_dict(
        as_state_dict(
            torch.load(weights, map_location="cpu", mmap=True, weights_only=False)
        )
    )
    if any("lora_" in name for name in state):
        raise ValueError(
            "Expected merged weights; export raw adapters with merge_lora first."
        )
    config_path = Path(training_config).expanduser().resolve()
    launch_path = Path(launch_metadata).expanduser().resolve()
    wandb = OmegaConf.load(config_path)
    # W&B wraps each top-level resolved configuration field in a value record.
    config = OmegaConf.create(
        {
            key: value["value"]
            for key, value in OmegaConf.to_container(wandb, resolve=True).items()
            if isinstance(value, dict) and "value" in value
        }
    )
    model = config.actor.model
    hierarchical = policy_kind == "explicit_subtask"
    if (
        bool(OmegaConf.select(model, "openpi.hierarchical.enabled", default=False))
        != hierarchical
    ):
        raise ValueError(
            f"Expected training hierarchy={hierarchical} for {policy_kind}."
        )
    sources = {"training_config": config_path, "launch_metadata": launch_path}
    if hierarchical:
        embedded_path = weights.parent / "openpi_training.json"
        metadata = json.loads(embedded_path.read_text())
        if metadata.get("text_template_version") != 1:
            raise ValueError(
                "Expected merged explicit subtask text_template_version=1."
            )
        trained = OmegaConf.create(metadata["model"])
        for field in DEPLOYMENT_FIELDS + (
            "openpi.hierarchical.enabled",
            "openpi.hierarchical.max_subtask_tokens",
        ):
            if OmegaConf.select(trained, field) != OmegaConf.select(model, field):
                raise ValueError(f"Expected matching checkpoint and W&B {field}.")
        normalization = metadata["normalization"]
        step = int(metadata["step"])
        sources["embedded_training_metadata"] = embedded_path
        provenance = "embedded_training_metadata_and_wandb"
    else:
        if bool(model.is_lora):
            raise ValueError("Expected Full Flow training with is_lora=false.")
        data_path = weights.parent.parent / "data.pt"
        saved = torch.load(data_path, map_location="cpu", weights_only=False)
        signatures = [rank["signature"] for rank in saved]
        norm_path = (
            Path(norm_stats_path or model.openpi_data.norm_stats_path)
            .expanduser()
            .resolve()
        )
        if not signatures or any(
            s["norm_sha256"] != file_sha256(norm_path) for s in signatures
        ):
            raise ValueError(
                "Expected normalization matching every saved DataLoader signature."
            )
        for signature in signatures:
            if signature["config_name"] != model.openpi.config_name:
                raise ValueError(
                    "Expected matching saved loader and training config_name."
                )
            if (
                signature["data_kwargs"].get("zero_base_action_targets", False)
                != model.openpi_data.zero_base_action_targets
            ):
                raise ValueError(
                    "Expected saved loader zero_base_action_targets matching training."
                )
            geometry = {
                "action_dim": model.openpi.model_action_dim,
                "action_horizon": model.num_action_chunks,
                "max_token_len": model.openpi.max_token_len,
                "discrete_state_input": model.openpi.discrete_state_input,
            }
            for key, expected in geometry.items():
                if signature["model"][key] != expected:
                    raise ValueError(f"Expected saved loader {key}={expected!r}.")
        normalization = json.loads(norm_path.read_text())
        matches = [re.fullmatch(r"global_step_(\d+)", part) for part in weights.parts]
        steps = [int(match[1]) for match in matches if match]
        if len(steps) != 1:
            raise ValueError(
                "Expected one global_step_<step> parent for Full Flow weights."
            )
        step = steps[0]
        sources.update(saved_loader=data_path, normalization=norm_path)
        provenance = "reconstructed_from_wandb_and_saved_loader"
    if (
        norm_stats_path
        and json.loads(Path(norm_stats_path).expanduser().read_text()) != normalization
    ):
        raise ValueError(
            "Expected supplied normalization identical to training records."
        )
    launch = json.loads(launch_path.read_text())
    manifest = {
        "format_version": 1,
        "policy_kind": policy_kind,
        "provenance": provenance,
        "step": step,
        "model": OmegaConf.to_container(model, resolve=True),
        "normalization": normalization,
        "text_template_version": 1 if hierarchical else None,
        "weights": {
            "path": str(weights),
            "size_bytes": weights.stat().st_size,
            "sha256": file_sha256(weights),
            "tensor_count": len(state),
        },
        "sources": {
            key: {"path": str(path), "sha256": file_sha256(path)}
            for key, path in sources.items()
        },
        "recorded_git_head": launch.get("git", {}).get("commit"),
        "historical_worktree_clean": "unknown",
    }
    destination.mkdir(parents=True)
    manifest_path = destination / "openpi_deployment.json"
    try:
        (destination / "norm_stats.json").write_text(
            json.dumps(normalization, indent=2) + "\n"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(destination)
        raise
    get_logger().info(
        "Prepared %s step %d provenance at %s", policy_kind, step, manifest_path
    )
    return manifest_path


def main() -> None:
    """Prepare a bundle without changing the source weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--policy-kind", choices=("flow_only", "explicit_subtask"), required=True
    )
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--launch-metadata", required=True)
    parser.add_argument("--norm-stats-path")
    args = parser.parse_args()
    prepare_deployment(
        args.ckpt,
        args.output,
        policy_kind=args.policy_kind,
        training_config=args.training_config,
        launch_metadata=args.launch_metadata,
        norm_stats_path=args.norm_stats_path,
    )


if __name__ == "__main__":
    main()
