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

"""Run a local robot_server against fake SDK and camera boundaries."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
from pathlib import Path

import numpy as np
from websockets.asyncio.server import serve


class SpiritaiSDK:
    """Record SDK dispatch and inject delay or post-ACK failure."""

    def __init__(self):
        self.rows = []
        self.fail = False
        self.delay_s = 0
        self.following = False

    def send_action_command(self, row):
        if self.fail:
            raise RuntimeError("injected SDK dispatch failure")
        time.sleep(self.delay_s)
        self.rows.append(row)

    send_joint_command = send_action_command

    def enable_external_following_mode(self):
        self.following = True

    def disable_external_following_mode(self):
        self.following = False

    def is_external_following_mode_enabled(self):
        return self.following


class SpiritaiContainer:
    """Serve the supplied native source with fake hardware, on loopback only."""

    def __init__(self, server_root: str | Path, *, state=None, images=None):
        sys.path.insert(0, str(Path(server_root) / "robot_server"))
        moz = importlib.import_module("mozrobot")
        native = importlib.import_module("server")
        self.sdk = SpiritaiSDK()
        self.sdk.get_robot_states = lambda: {
            "left_arm_ee_pos": state[:6],
            "left_arm_joints": np.zeros(7),
            "left_arm_psi": state[6],
            "left_gripper_position": state[7],
            "right_arm_ee_pos": state[8:14],
            "right_arm_joints": np.zeros(7),
            "right_arm_psi": state[14],
            "right_gripper_position": state[15],
            "torso_ee_pos": state[16:22],
            "torso_joints": np.zeros(6),
            "base_speed": state[22:25],
        }

        class Cameras:
            def get_images(self):
                return images

        self._cameras = Cameras()
        self.robot = moz.MozRobot(moz.MozRobotConfig(no_camera=True))
        self.robot._recorder = self._cameras
        self.robot.structure = "whole_body"
        self.robot._robot = self.sdk
        self.robot.cart_dim = 23
        self.robot.joint_dim = 25
        self.server = native.RobotServer(self.robot, max_chunk=60)
        self.ready = threading.Event()

    def __enter__(self):
        async def run():
            self.loop = asyncio.get_running_loop()
            self.stop = asyncio.Event()
            async with serve(
                self.server._handler, "127.0.0.1", 0, compression=None, max_size=None
            ) as listener:
                self.url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
                self.ready.set()
                await self.stop.wait()

        self.thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
        self.thread.start()
        assert self.ready.wait(5)
        return self

    def __exit__(self, *args):
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(5)
        assert not self.thread.is_alive()
