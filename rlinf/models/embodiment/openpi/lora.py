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

"""Native adapters for both OpenPI Gemma experts and deploy weight merging."""

from __future__ import annotations

import torch
from torch import nn

from rlinf.models.embodiment.openpi.modules.lora import (
    FeedForward,
    Linear,
    LoRAConfig,
)


def configure_lora(
    model: nn.Module,
    *,
    vlm_rank: int,
    expert_rank: int,
    vlm_alpha: float,
    expert_alpha: float,
) -> dict[str, int]:
    """Freeze the base and adapt attention/FFN in both experts after loading."""
    configs = [LoRAConfig(vlm_rank, vlm_alpha), LoRAConfig(expert_rank, expert_alpha)]
    if getattr(model, "native_lora", False):
        raise ValueError("OpenPI native LoRA is already configured.")
    model.requires_grad_(False)
    for block in model.llm.layers:
        for index, config in enumerate(configs):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                projections = getattr(block.attn, name)
                if projections[index] is not None:
                    projections[index] = Linear.from_linear(projections[index], config)
            block.mlps[index] = FeedForward.from_base(block.mlps[index], config)
    model.native_lora = True
    counts = {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    model.lora_parameter_counts = counts
    return counts


def merge_lora_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merge native adapters into bare Pi0 keys without mutating the input."""
    merged = dict(state)
    for name, scaling in state.items():
        if not name.endswith(".lora_scaling"):
            continue
        prefix = name[: -len("lora_scaling")]
        if prefix + "lora_a" in state:
            pairs = [("weight", "lora_b", "lora_a")]
        else:
            pairs = [
                ("w_gating", "w_gating_lora_a", "w_gating_lora_b"),
                ("w_linear", "w_linear_lora_a", "w_linear_lora_b"),
            ]
        for base_name, left, right in pairs:
            base, a, b = (prefix + key for key in (base_name, left, right))
            if any(key not in state for key in (base, a, b)):
                raise ValueError(f"Incomplete OpenPI LoRA weights under {prefix!r}.")
            delta = torch.matmul(state[a].float(), state[b].float()) * scaling.float()
            merged[base] = (state[base].float() + delta).to(state[base].dtype)
            del merged[a], merged[b]
        del merged[name]
    remaining = [name for name in merged if "lora_" in name]
    if remaining:
        raise ValueError(f"Unmerged OpenPI LoRA parameters: {remaining[:5]}.")
    return merged
