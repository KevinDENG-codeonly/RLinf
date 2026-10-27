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

"""Loopback implementation of the SpiritAI container protocol for CPU tests."""

from __future__ import annotations

import threading

import msgpack
import numpy as np
from websockets.sync.server import serve


class SpiritaiServer:
    """Serve measured observations and record accepted Cartesian commands."""

    def __init__(self):
        self.commands = []
        self.messages = []
        self.following = False
        self.reject_commands = False
        self.stall_observations = False
        self.stopped = threading.Event()
        self.obs = {
            key: np.full(dim, index / 10, dtype=np.float32)
            for index, (key, dim) in enumerate(
                (
                    ("leftarm_state_cart_pos", 6),
                    ("leftarm_state_psi", 1),
                    ("leftarm_gripper_state_pos", 1),
                    ("rightarm_state_cart_pos", 6),
                    ("rightarm_state_psi", 1),
                    ("rightarm_gripper_state_pos", 1),
                    ("torso_state_cart_pos", 6),
                    ("base_state_speed", 3),
                )
            )
        }
        self.images = {
            key: np.full((4, 5, 3), value, dtype=np.uint8)
            for key, value in (
                ("cam_high", 10),
                ("cam_left_wrist", 20),
                ("cam_right_wrist", 30),
            )
        }

    @staticmethod
    def _encode(value):
        if isinstance(value, np.ndarray):
            return {
                b"nd": True,
                b"type": value.dtype.str,
                b"kind": b"",
                b"shape": value.shape,
                b"data": value.tobytes(),
            }
        raise TypeError(type(value))

    @staticmethod
    def _decode(value):
        if b"nd" in value:
            return np.frombuffer(value[b"data"], dtype=value[b"type"]).reshape(
                value[b"shape"]
            )
        return value

    def _serve(self, connection):
        connection.send(
            msgpack.packb(
                {
                    "type": "hello",
                    "metadata": {
                        "cart_dim": 23,
                        "accepted_cart_dims": [14, 20, 23],
                        "max_chunk": 60,
                    },
                },
                use_bin_type=True,
            )
        )
        for raw in connection:
            request = msgpack.unpackb(raw, raw=False, object_hook=self._decode)
            self.messages.append(request)
            kind = request["type"]
            if kind == "get_obs":
                if self.stall_observations:
                    self.stopped.wait(2)
                    return
                response = {"type": "obs", "obs": self.obs, "images": self.images}
            elif kind == "get_status":
                response = {"type": "status", "busy": False}
            elif kind == "set_external_following":
                self.following = request["enabled"]
                response = {
                    "type": "external_following",
                    "accepted": True,
                    "enabled": self.following,
                }
            elif kind == "send_command":
                self.commands.append(request)
                response = {"type": "ack", "accepted": not self.reject_commands}
            else:
                response = {"type": "error", "message": kind}
            connection.send(
                msgpack.packb(response, default=self._encode, use_bin_type=True)
            )

    def __enter__(self):
        self.server = serve(self._serve, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.server.shutdown()
        self.thread.join(timeout=3)
