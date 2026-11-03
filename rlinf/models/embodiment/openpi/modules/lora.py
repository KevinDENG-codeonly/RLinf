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

"""LoRA (Low-Rank Adaptation) modules for PyTorch, aligned with JAX models/lora.py."""

from __future__ import annotations

import dataclasses
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import gelu_glu


@dataclasses.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    rank: int
    alpha: float = 1.0
    rslora: bool = False
    axes: tuple[int, int] = (-2, -1)
    label: str = "L"

    def __post_init__(self):
        if self.rank <= 0 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA requires a positive rank and finite positive alpha.")

    @property
    def scaling_value(self) -> float:
        return (
            self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank
        )


class Linear(nn.Linear):
    """Linear adapter retaining the original weight and bias checkpoint keys."""

    @classmethod
    def from_linear(cls, base: nn.Linear, config: LoRAConfig) -> Linear:
        """Attach zero-output adapters to an already loaded linear layer."""
        result = cls(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        result.weight = base.weight
        result.bias = base.bias
        result.lora_a = nn.Parameter(
            base.weight.new_empty(config.rank, base.in_features)
        )
        result.lora_b = nn.Parameter(
            base.weight.new_zeros(base.out_features, config.rank)
        )
        nn.init.normal_(result.lora_a, std=0.01)
        result.register_buffer(
            "lora_scaling", base.weight.new_tensor(config.scaling_value)
        )
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.linear(x, self.weight, self.bias)
            + F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
            * self.lora_scaling
        )


class Einsum(nn.Module):
    """Einsum with LoRA support. Drop-in replacement for a linear einsum operation.

    Follows the JAX convention of storing weight shape explicitly and
    computing via einsum rather than standard matmul.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        eqn: str,
        init_fn: str = "lecun_normal",
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__()
        self.shape = shape
        self.eqn = eqn
        self.lora_config = lora_config

        # Initialize weight
        w = torch.empty(*shape)
        if init_fn == "lecun_normal":
            # lecun_normal: std = sqrt(1 / fan_in)
            # Weight shape is typically (out_features, in_features) or (num_heads, width, head_dim)
            # For the weight tensor, fan_in is the last dimension size
            nn.init.normal_(w, std=1.0 / math.sqrt(shape[-1]))
        elif init_fn == "zeros":
            nn.init.zeros_(w)
        elif init_fn == "normal":
            nn.init.normal_(w, std=0.01)
        else:
            nn.init.normal_(w, std=0.01)
        self.w = nn.Parameter(w)

        # LoRA parameters
        if lora_config is not None:
            axes = lora_config.axes
            rank = lora_config.rank
            shape_a = list(shape)
            shape_a[axes[1]] = rank
            shape_b = list(shape)
            shape_b[axes[0]] = rank

            w_a = torch.empty(*shape_a)
            w_b = torch.zeros(*shape_b)  # LoRA B initialized to zero
            nn.init.normal_(w_a, std=0.01)
            self.w_a = nn.Parameter(w_a)
            self.w_b = nn.Parameter(w_b)
            self._eqn_a, self._eqn_b = self._make_lora_eqns(
                eqn, axes, lora_config.label
            )
        else:
            self.w_a = None
            self.w_b = None

    def _make_lora_eqns(
        self, eqn: str, axes: tuple[int, int], label: str
    ) -> tuple[str, str]:
        """Create einsum equations for LoRA computation."""
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        m = re.match(r"(.*),(.*)->(.*)", eqn)
        if not m:
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        a_label, b_label = rhs[axes[0]], rhs[axes[1]]

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b

    def forward(self, x: torch.Tensor, eqn: str | None = None) -> torch.Tensor:
        """Forward pass with optional LoRA."""
        if eqn is None:
            eqn = self.eqn

        # Parse equation: "input_spec,weight_spec->output_spec"
        result = torch.einsum(eqn, x, self.w.to(x.dtype))

        if self.lora_config is not None:
            lora = torch.einsum(self._eqn_a, x, self.w_a.to(x.dtype))
            lora = torch.einsum(self._eqn_b, lora, self.w_b.to(x.dtype))
            result = result + lora * self.lora_config.scaling_value

        return result


class FeedForward(nn.Module):
    """FeedForward module with optional LoRA support.

    Uses gated FFN: gate(x) * linear(x) -> output.
    """

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.lora_config = lora_config

        # Gating weights: shape (2, features, hidden_dim)
        w_gating = torch.empty(2, features, hidden_dim)
        nn.init.normal_(w_gating, std=1.0 / math.sqrt(features))
        self.w_gating = nn.Parameter(w_gating)

        # Linear weights: shape (hidden_dim, features)
        w_linear = torch.empty(hidden_dim, features)
        nn.init.normal_(w_linear, std=1.0 / math.sqrt(hidden_dim))
        self.w_linear = nn.Parameter(w_linear)

        self._init_adapters()

    @classmethod
    def from_base(cls, base: nn.Module, config: LoRAConfig) -> FeedForward:
        """Reuse loaded FFN parameters without allocating another full FFN."""
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.features = base.features
        result.hidden_dim = base.hidden_dim
        result.w_gating = base.w_gating
        result.w_linear = base.w_linear
        result.lora_config = config
        result._init_adapters()
        return result

    def _init_adapters(self) -> None:
        if self.lora_config is not None:
            rank = self.lora_config.rank
            features, hidden_dim = self.features, self.hidden_dim
            # LoRA for w_gating[0] and w_gating[1] separately
            self.w_gating_lora_a = nn.Parameter(
                self.w_gating.new_empty(2, features, rank)
            )
            self.w_gating_lora_b = nn.Parameter(
                self.w_gating.new_zeros(2, rank, hidden_dim)
            )
            self.w_linear_lora_a = nn.Parameter(
                self.w_linear.new_empty(hidden_dim, rank)
            )
            self.w_linear_lora_b = nn.Parameter(self.w_linear.new_zeros(rank, features))
            nn.init.normal_(self.w_gating_lora_a, std=0.01)
            nn.init.normal_(self.w_linear_lora_a, std=0.01)
            self.register_buffer(
                "lora_scaling", self.w_linear.new_tensor(self.lora_config.scaling_value)
            )
        else:
            self.w_gating_lora_a = None
            self.w_gating_lora_b = None
            self.w_linear_lora_a = None
            self.w_linear_lora_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gating and optional LoRA."""
        # Gate
        ff_gate = self._dot(x, self.w_gating[0], 0)

        # FF1
        ff1 = self._dot(x, self.w_gating[1], 1)
        activations = gelu_glu(ff_gate, ff1)

        # Output
        outputs = self._dot(activations, self.w_linear, -1)
        return outputs

    def _dot(self, x: torch.Tensor, w: torch.Tensor, index: int) -> torch.Tensor:
        """Dot product with optional LoRA."""
        base = torch.matmul(x, w.to(x.dtype))
        if self.lora_config is not None:
            scaling = self.lora_scaling
            if index == 0:
                lora = torch.matmul(
                    torch.matmul(x, self.w_gating_lora_a[0].to(x.dtype)),
                    self.w_gating_lora_b[0].to(x.dtype),
                )
            elif index == 1:
                lora = torch.matmul(
                    torch.matmul(x, self.w_gating_lora_a[1].to(x.dtype)),
                    self.w_gating_lora_b[1].to(x.dtype),
                )
            else:  # linear output
                lora = torch.matmul(
                    torch.matmul(x, self.w_linear_lora_a.to(x.dtype)),
                    self.w_linear_lora_b.to(x.dtype),
                )
            base = base + lora * scaling
        return base
