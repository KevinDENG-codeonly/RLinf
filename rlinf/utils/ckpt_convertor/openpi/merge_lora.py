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

"""Merge native Pi0.5 LoRA checkpoints for RLinf hierarchical evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rlinf.models.embodiment.openpi.lora import merge_lora_state_dict
from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict, strip_wrapper_prefix
from rlinf.utils.ckpt_convertor.openpi.sft2deploy import resolve_full_weights
from rlinf.utils.logging import get_logger

logger = get_logger()


def export_lora_checkpoint(checkpoint: str | Path, output: str | Path) -> Path:
    """Export merged native weights with training settings and normalization."""
    weights = resolve_full_weights(checkpoint)
    metadata_path = weights.parent.parent / "openpi_training.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expected hierarchical checkpoint metadata at {metadata_path}."
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("text_template_version") != 1:
        raise ValueError("Expected hierarchical text_template_version=1.")
    output = Path(output)
    output_weights = output / "full_weights.pt"
    if output_weights.exists():
        raise FileExistsError(
            f"Refusing to overwrite deploy weights at {output_weights}."
        )
    state = strip_wrapper_prefix(
        as_state_dict(
            torch.load(weights, map_location="cpu", mmap=True, weights_only=False)
        ),
        cast_dtype=None,
    )
    if not any(name.endswith(".lora_scaling") for name in state):
        raise ValueError(
            "Expected native OpenPI LoRA parameters in the training checkpoint."
        )
    merged = merge_lora_state_dict(state)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_weights)
    (output / "norm_stats.json").write_text(
        json.dumps(metadata["normalization"], indent=2) + "\n"
    )
    (output / "openpi_training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info(
        "Merged native OpenPI LoRA -> %s (%d tensors)", output_weights, len(merged)
    )
    return output_weights


def main() -> None:
    """Run the standalone merge/export command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_lora_checkpoint(args.ckpt, args.output)


if __name__ == "__main__":
    main()
