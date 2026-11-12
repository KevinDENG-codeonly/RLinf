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

"""Requester-owned deadlines and action timelines for one RTC episode."""

import math
import uuid
from collections import deque
from typing import Any

import torch

from rlinf.data.schema.embodied_types import RTCActionResponse, RTCRequest


class RTCSession:
    """Own one in-flight request and a wall-clock action chunk.

    ``now`` values come from the env worker's monotonic clock. Timestamps
    from the rollout host or robot are never compared with this clock.
    """

    def __init__(
        self,
        source_hz: float,
        request_timeout_s: float,
        bootstrap_timeout_s: float,
        min_exec_horizon: int,
        initial_delay_steps: int,
        delay_buffer_size: int,
    ) -> None:
        for name, value in (
            ("source_hz", source_hz),
            ("request_timeout_s", request_timeout_s),
            ("bootstrap_timeout_s", bootstrap_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Expected positive finite {name}, got {value}.")
        if min_exec_horizon <= 0 or initial_delay_steps < 0 or delay_buffer_size <= 0:
            raise ValueError(
                "Expected positive execution horizon/buffer and nonnegative delay."
            )
        self.source_hz = source_hz
        self.request_timeout_s = request_timeout_s
        self.bootstrap_timeout_s = bootstrap_timeout_s
        self.min_exec_horizon = min_exec_horizon
        self.session_id = uuid.uuid4().hex
        self.pending: RTCRequest | None = None
        self.requested_at = 0.0
        self.chunk_started_at = 0.0
        self.actions: torch.Tensor | None = None
        self.response: RTCActionResponse | None = None
        self._next_chunk_id = 0
        self._next_action_index = 0
        self._delays = deque([initial_delay_steps], maxlen=delay_buffer_size)

    def begin(self, obs: dict[str, Any], request_type: str, now: float) -> RTCRequest:
        """Begin bootstrap, warmup or replan; parallel requests are rejected."""
        if self.pending is not None:
            raise RuntimeError("Expected at most one in-flight RTC request.")
        if request_type not in ("bootstrap", "warmup", "replan"):
            raise ValueError(
                f"Expected an inference request type, got {request_type!r}."
            )
        request = RTCRequest(
            obs=obs,
            request_type=request_type,
            session_id=self.session_id,
            chunk_id=self._next_chunk_id,
            executed_horizon=self.action_index(now) if request_type == "replan" else 0,
            predicted_delay_steps=max(self._delays),
        )
        self._next_chunk_id += 1
        self.pending = request
        self.requested_at = now
        return request

    def check_deadline(self, now: float) -> None:
        """Raise when the pending request exceeds its local wall-clock budget."""
        if self.pending is None:
            return
        timeout = (
            self.request_timeout_s
            if self.pending.request_type == "replan"
            else self.bootstrap_timeout_s
        )
        if now - self.requested_at >= timeout:
            raise TimeoutError(
                f"RTC {self.pending.request_type} request {self.pending.chunk_id} "
                f"exceeded {timeout:.3f}s; late actions must be discarded."
            )

    def accept(self, response: RTCActionResponse, now: float) -> None:
        """Validate and install a timely chunk, skipping elapsed replan ticks."""
        if self.pending is None:
            raise RuntimeError(
                "Expected an in-flight RTC request before accepting actions."
            )
        self.check_deadline(now)
        self.pending.validate_response(response)
        actions = response.actions
        if (
            actions.ndim != 3
            or actions.shape[0] != 1
            or actions.shape[1] <= self.min_exec_horizon
        ):
            raise ValueError(
                "Expected a single-env RTC chunk longer than min_exec_horizon."
            )
        if self.pending.request_type != "warmup":
            started_at = (
                self.requested_at if self.pending.request_type == "replan" else now
            )
            if (
                math.floor(max(0.0, now - started_at) * self.source_hz)
                >= actions.shape[1]
            ):
                raise TimeoutError(
                    "Expected unexpired RTC actions after inference latency."
                )
            self.chunk_started_at = started_at
            self._next_action_index = 0
            self.actions = actions
            self.response = response
        if self.pending.request_type == "replan":
            self._delays.append(math.ceil((now - self.requested_at) * self.source_hz))
        self.pending = None

    def action_index(self, now: float) -> int:
        """Return the next row, accounting for RPC overruns and sent actions."""
        return max(
            self._next_action_index,
            math.floor(max(0.0, now - self.chunk_started_at) * self.source_hz),
        )

    def take_action(self, now: float) -> torch.Tensor | None:
        """Consume the next target, or return None when the chunk is exhausted."""
        if self.actions is None:
            return None
        index = self.action_index(now)
        if index >= self.actions.shape[1]:
            return None
        self._next_action_index = index + 1
        return self.actions[:, index]

    def should_replan(self, now: float) -> bool:
        """Request a replacement after the minimum execution horizon."""
        return (
            self.pending is None
            and self.actions is not None
            and self.action_index(now) >= self.min_exec_horizon
        )
