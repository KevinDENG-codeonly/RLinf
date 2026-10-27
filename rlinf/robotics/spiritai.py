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

"""Client for an existing SpiritAI robot_server WebSocket container."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = (
    "leftarm_state_cart_pos",
    "leftarm_state_psi",
    "leftarm_gripper_state_pos",
    "rightarm_state_cart_pos",
    "rightarm_state_psi",
    "rightarm_gripper_state_pos",
    "torso_state_cart_pos",
    "base_state_speed",
)
ACTION_KEYS = tuple(key.replace("_state_", "_cmd_") for key in STATE_KEYS)
FIELD_DIMS = (6, 1, 1, 6, 1, 1, 6, 3)
ACTION_DIM = 25


def pack_cartesian_fields(data: dict, keys: tuple[str, ...]) -> np.ndarray:
    """Concatenate named SpiritAI fields in the policy's Cartesian order."""
    fields = [np.asarray(data[key], dtype=np.float32) for key in keys]
    for key, value, dim in zip(keys, fields, FIELD_DIMS, strict=True):
        if value.ndim < 1 or value.shape[-1] != dim:
            raise ValueError(f"Expected {key} with last dim {dim}, got {value.shape}.")
    result = np.concatenate(fields, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError(f"Expected finite Cartesian fields {keys}.")
    return result


def _encode_numpy(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Expected a numeric robot_server array, got object dtype.")
        result = {
            b"nd": isinstance(value, np.ndarray),
            b"type": array.dtype.str,
            b"data": array.tobytes(),
        }
        if isinstance(value, np.ndarray):
            result.update({b"kind": b"", b"shape": array.shape})
        return result
    raise TypeError(f"Unsupported robot_server value: {type(value).__name__}.")


def _decode_numpy(value: dict) -> Any:
    if b"nd" not in value:
        return value
    dtype = np.dtype(value[b"type"])
    if dtype.hasobject:
        raise ValueError("Expected a numeric robot_server array, got object dtype.")
    array = np.frombuffer(value[b"data"], dtype=dtype)
    return array.reshape(value[b"shape"]) if value[b"nd"] else array[0]


class SpiritaiClient:
    """Own one serialized, timeout-bounded robot_server connection.

    Construction does not connect. The caller closes the session; a failed
    request closes it before another request can consume a stale reply.
    """

    def __init__(self, url: str, timeout_s: float = 2.0) -> None:
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(f"Expected positive finite timeout_s, got {timeout_s}.")
        self.url = url
        self.timeout_s = timeout_s
        self.metadata: dict[str, Any] = {}
        self._runner: asyncio.Runner | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._ws: Any = None

    async def _receive(self) -> dict[str, Any]:
        import msgpack

        message = await self._ws.recv()
        result = msgpack.unpackb(message, object_hook=_decode_numpy, raw=False)
        if not isinstance(result, dict) or result.get("type") == "error":
            raise RuntimeError(f"Expected robot_server response, got {result!r}.")
        return result

    def connect(self) -> None:
        """Connect and validate Cartesian command capabilities."""
        if self._ws is not None:
            return
        from websockets.asyncio.client import connect

        async def open_session() -> dict[str, Any]:
            self._ws = await connect(
                self.url,
                max_size=None,
                compression=None,
                ping_interval=None,
                proxy=None,
                open_timeout=self.timeout_s,
                close_timeout=self.timeout_s,
            )
            return await self._receive()

        self._runner = asyncio.Runner()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="SpiritAI"
        )
        try:
            hello = self._executor.submit(
                self._runner.run, asyncio.wait_for(open_session(), self.timeout_s)
            ).result()
            if hello.get("type") != "hello":
                raise RuntimeError(f"Expected robot_server hello, got {hello!r}.")
            self.metadata = dict(hello["metadata"])
            self.command_dim
        except BaseException:
            self.close()
            raise

    @property
    def command_dim(self) -> int:
        """Return the widest Cartesian layout accepted by the container."""
        maximum = int(self.metadata.get("cart_dim", 0))
        accepted = self.metadata.get("accepted_cart_dims", ())
        for dim in (23, 20, 14):
            if dim <= maximum and (not accepted or dim in accepted):
                return dim
        raise ValueError(
            f"Expected Cartesian robot_server capabilities, got {self.metadata}."
        )

    def request(
        self, message: dict[str, Any], *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Send one request and consume its response within a total deadline."""
        import msgpack

        if self._ws is None or self._runner is None:
            raise RuntimeError("Expected a connected SpiritAI client before request().")
        timeout = self.timeout_s if timeout_s is None else timeout_s

        async def exchange() -> dict[str, Any]:
            await self._ws.send(
                msgpack.packb(message, default=_encode_numpy, use_bin_type=True)
            )
            return await self._receive()

        try:
            return self._executor.submit(
                self._runner.run, asyncio.wait_for(exchange(), timeout)
            ).result()
        except BaseException:
            self.close()
            raise

    @property
    def connected(self) -> bool:
        """Return whether this client owns an open session."""
        return self._ws is not None

    def observation(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Read the container's measured state and decoded RGB frames."""
        message = self.request({"type": "get_obs"})
        if message.get("type", "obs") != "obs":
            raise RuntimeError(f"Expected robot_server obs, got {message!r}.")
        return message["obs"], message["images"]

    def wait_idle(self) -> None:
        """Wait at most timeout_s for the accepted command to finish."""
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Expected idle SpiritAI robot before the command deadline."
                )
            if not self.request({"type": "get_status"}, timeout_s=remaining)["busy"]:
                return
            time.sleep(min(0.001, remaining))

    def send_actions(self, actions: np.ndarray, source_hz: float) -> None:
        """Send an (H, 25) policy chunk using the container's Cartesian layout."""
        actions = np.asarray(actions, dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[-1] != 25
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                f"Expected finite SpiritAI actions (H, 25), got {actions.shape}."
            )
        if not np.isfinite(source_hz) or source_hz <= 0:
            raise ValueError(f"Expected positive finite source_hz, got {source_hz}.")
        maximum = int(self.metadata.get("max_chunk", 60))
        if not 0 < len(actions) <= maximum:
            raise ValueError(f"Expected 1..{maximum} action rows, got {len(actions)}.")
        dim = self.command_dim
        indices = [*range(6), 7, *range(8, 14), 15]
        if dim >= 20:
            indices.extend(range(16, 22))
        if dim == 23:
            indices.extend(range(22, 25))
        self.wait_idle()
        response = self.request(
            {
                "type": "send_command",
                "kind": "cart",
                "actions": actions[:, indices],
                "source_hz": source_hz,
            }
        )
        if not response.get("accepted", False):
            raise RuntimeError(f"Expected accepted SpiritAI command, got {response!r}.")

    def close(self) -> None:
        """Close the session; repeated calls are harmless."""
        runner, executor, ws = self._runner, self._executor, self._ws
        self._runner = None
        self._executor = None
        self._ws = None

        def shutdown() -> None:
            try:
                if ws is not None:
                    runner.run(ws.close())
            finally:
                runner.close()

        if executor is not None:
            try:
                executor.submit(shutdown).result()
            finally:
                executor.shutdown(wait=True)
