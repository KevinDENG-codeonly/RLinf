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

"""Explicit subtask configuration and shared training/inference tokenization."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class HierarchicalConfig:
    """Settings for alternating subtask CE and subtask-conditioned flow loss."""

    enabled: bool = False
    text_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0
    max_subtask_tokens: int = 96
    planning_interval: int = 1
    ce_chunk_size: int = 32

    def __post_init__(self):
        for name in ("text_loss_weight", "flow_loss_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Expected positive finite {name}, got {value}.")
        for name in ("max_subtask_tokens", "planning_interval", "ce_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Expected positive integer {name}, got {value!r}.")

    @classmethod
    def from_mapping(cls, value: Any = None) -> HierarchicalConfig:
        """Read optional Hydra settings without modifying the supplied config."""
        return cls(**dict(value or {}))


@dataclasses.dataclass
class SubtaskPlanningState:
    """Caller-owned planning cache; reset at every environment episode boundary."""

    instructions: tuple[str, ...] | None = None
    subtasks: list[str] | None = None
    action_chunks: int = 0

    def reset(self) -> None:
        """Invalidate cached text even when the next episode has the same task."""
        self.instructions = None
        self.subtasks = None
        self.action_chunks = 0

    def should_plan(self, instructions: list[str], interval: int) -> bool:
        """Refresh on a new instruction, an empty cache, or the configured cadence."""
        return (
            self.subtasks is None
            or self.instructions != tuple(instructions)
            or self.action_chunks >= interval
        )

    def update(self, instructions: list[str], subtasks: list[str]) -> None:
        """Commit one generated subtask per environment in the caller's batch."""
        if len(instructions) != len(subtasks) or any(
            not text.strip() for text in subtasks
        ):
            raise ValueError("Planning requires one nonempty subtask per environment.")
        self.instructions = tuple(instructions)
        self.subtasks = list(subtasks)
        self.action_chunks = 0


class SubtaskTokenizer:
    """PaliGemma text prefix and causal subtask response, including EOS."""

    def __init__(
        self, max_length: int, max_subtask_tokens: int = 96, processor: Any = None
    ):
        self.max_length = max_length
        self.max_subtask_tokens = max_subtask_tokens
        if processor is None:
            from openpi.models.tokenizer import PaligemmaTokenizer

            processor = PaligemmaTokenizer(max_length)._tokenizer
        self.processor = processor

    def encode(
        self,
        overall: str,
        subtask: str | None = None,
        state: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Build a bidirectional instruction prefix and a causal response span."""
        if not isinstance(overall, str) or not overall.strip():
            raise ValueError(
                "High-level prediction requires a nonempty overall instruction."
            )
        overall = overall.strip().replace("_", " ").replace("\n", " ")
        prefix = f"Task: {overall}"
        if state is not None:
            state_bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            prefix += f", State: {' '.join(map(str, state_bins))};"
        prefix += "\nSubtask: "
        prefix_ids = self.processor.encode(prefix, add_bos=True)
        response_ids = []
        if subtask is not None:
            if not isinstance(subtask, str) or not subtask.strip():
                raise ValueError("High-level supervision requires a nonempty subtask.")
            response_ids = self.processor.encode(subtask.strip(), add_eos=True)
            if len(response_ids) > self.max_subtask_tokens:
                raise ValueError(
                    f"Subtask needs {len(response_ids)} tokens including EOS; "
                    f"max_subtask_tokens={self.max_subtask_tokens}."
                )
        length = len(prefix_ids) + len(response_ids)
        if length > self.max_length:
            raise ValueError(
                f"High-level sequence needs {length} tokens; max_token_len="
                f"{self.max_length}. Increase the configured length; targets are never truncated."
            )
        ids = np.zeros(self.max_length, dtype=np.int64)
        valid = np.zeros(self.max_length, dtype=np.bool_)
        response = np.zeros(self.max_length, dtype=np.bool_)
        ids[:length] = prefix_ids + response_ids
        valid[:length] = True
        response[len(prefix_ids) : length] = True
        return {
            "tokenized_prompt": ids,
            "tokenized_prompt_mask": valid,
            "token_ar_mask": response,
            "token_loss_mask": response.copy(),
        }

    def decode(self, ids: list[int]) -> str:
        """Decode the response before its first EOS, excluding special tokens."""
        eos = self.processor.eos_id()
        if eos in ids:
            ids = ids[: ids.index(eos)]
        return self.processor.decode(ids).strip()


@dataclasses.dataclass(frozen=True)
class HierarchicalInputs:
    """Retain the task as the HL instruction and use subtask for the LL prefix."""

    transform: Any

    def __call__(self, data: dict) -> dict:
        subtask = data.get("subtask")
        if not isinstance(subtask, str) or not subtask.strip():
            raise ValueError(
                "Hierarchical SFT requires a nonempty per-frame subtask column."
            )
        result = self.transform(data)
        overall = result.get("prompt")
        if not isinstance(overall, str) or not overall.strip():
            raise ValueError(
                "Hierarchical SFT requires overall instruction from the LeRobot task."
            )
        result["overall_instruction"] = overall
        result["subtask"] = subtask
        result["prompt"] = subtask
        return result


@dataclasses.dataclass(frozen=True)
class TokenizeHighLevel:
    """Tokenize HL supervision after state normalization and before LL tokenization."""

    tokenizer: SubtaskTokenizer
    discrete_state_input: bool

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        state = data["state"] if self.discrete_state_input else None
        tokens = self.tokenizer.encode(
            data.pop("overall_instruction"), data.pop("subtask"), state
        )
        data.update({f"high_level_{key}": value for key, value in tokens.items()})
        return data
