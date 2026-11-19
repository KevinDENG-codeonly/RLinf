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

"""Measured observations and Cartesian control through SpiritAI robot_server."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.robotics.spiritai import (
    CAMERA_KEYS,
    STATE_KEYS,
    SpiritaiClient,
    pack_cartesian_fields,
)


@dataclass
class SpiritaiCartesianConfig:
    """Connection and task settings for an existing robot container."""

    robot_url: str = "ws://127.0.0.1:8766"
    timeout_s: float = 2.0
    source_hz: float = 30.0
    task_description: str = ""
    image_height: int = 480
    image_width: int = 640
    zero_base_actions: bool = False
    enable_external_following: bool = False
    execution_enabled: bool = True
    defer_external_following: bool = False
    require_execution_capabilities: bool = False
    min_cartesian_dim: int = 14
    gripper_min: float | None = None
    gripper_max: float | None = None
    max_arm_translation_m_s: float = 0.08
    max_arm_rotation_rad_s: float = 0.35
    max_torso_translation_m_s: float = 0.04
    max_torso_rotation_rad_s: float = 0.2
    max_gripper_velocity_s: float = 0.8
    max_base_speed: float = 0.05

    def __post_init__(self) -> None:
        if self.min_cartesian_dim not in (14, 20, 23):
            raise ValueError(
                f"Expected min_cartesian_dim in (14, 20, 23), got {self.min_cartesian_dim}."
            )
        if not np.isfinite(self.source_hz) or self.source_hz <= 0:
            raise ValueError(
                f"Expected positive finite source_hz, got {self.source_hz}."
            )
        for name in (
            "max_arm_translation_m_s",
            "max_arm_rotation_rad_s",
            "max_torso_translation_m_s",
            "max_torso_rotation_rad_s",
            "max_gripper_velocity_s",
            "max_base_speed",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Expected finite nonnegative {name}, got {value}.")
        if (self.gripper_min is None) != (self.gripper_max is None):
            raise ValueError("Expected both gripper_min and gripper_max, or neither.")
        if self.gripper_min is not None and not (
            np.isfinite(self.gripper_min)
            and np.isfinite(self.gripper_max)
            and self.gripper_min < self.gripper_max
        ):
            raise ValueError("Expected finite ordered gripper command bounds.")


class SpiritaiCartesianEnv(gym.Env):
    """Send one action row per RTC tick to a real robot container.

    Reset reads the current pose without moving to a preset pose. The
    container owns low-level control; this env owns the WebSocket client.
    """

    def __init__(
        self,
        override_cfg: dict[str, Any] | None = None,
        worker_info: Any = None,
        robot_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        super().__init__()
        self.config = SpiritaiCartesianConfig(**dict(override_cfg or {}))
        self.task_description = self.config.task_description
        self.client = SpiritaiClient(self.config.robot_url, self.config.timeout_s)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (25,), np.float32)

        def image_space() -> gym.spaces.Box:
            return gym.spaces.Box(
                0, 255, (self.config.image_height, self.config.image_width, 3), np.uint8
            )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict({"cartesian": self.action_space}),
                "frames": gym.spaces.Dict({key: image_space() for key in CAMERA_KEYS}),
            }
        )
        self._state: np.ndarray | None = None
        self._following_enabled = False

    def _observe(self) -> dict[str, Any]:
        obs, images = self.client.observation()
        self._state = pack_cartesian_fields(obs, STATE_KEYS)
        frames = {key: np.asarray(images[key]) for key in CAMERA_KEYS}
        result = {"state": {"cartesian": self._state.copy()}, "frames": frames}
        if not self.observation_space.contains(result):
            raise ValueError(
                f"Expected SpiritAI RGB uint8 images "
                f"{(self.config.image_height, self.config.image_width, 3)}, got "
                f"{ {key: (frame.shape, str(frame.dtype)) for key, frame in frames.items()} }."
            )
        return result

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Connect and read the physical robot's current observation."""
        super().reset(seed=seed)
        try:
            self.client.connect()
            if self.client.command_dim < self.config.min_cartesian_dim:
                raise ValueError(
                    f"Expected Cartesian dimension >= {self.config.min_cartesian_dim}, "
                    f"got {self.client.command_dim}; torso commands would be lost."
                )
            self.client.wait_idle()
            observation = self._observe()
            if self.config.execution_enabled:
                if self.config.require_execution_capabilities:
                    self.client.validate_execution_capabilities()
                if not self.config.defer_external_following:
                    self._enable_following()
            return observation, {}
        except BaseException:
            self.close()
            raise

    def _enable_following(self) -> None:
        if self.config.enable_external_following and not self._following_enabled:
            response = self.client.request(
                {"type": "set_external_following", "enabled": True}
            )
            if not response.get("accepted") or response.get("enabled") is not True:
                raise RuntimeError(
                    f"Expected enabled external following, got {response!r}."
                )
            self._following_enabled = True

    def get_hold_action(self, fallback_action: Any = None) -> np.ndarray:
        """Hold measured pose and grippers with zero base velocity."""
        if self._state is None:
            raise RuntimeError("Expected reset() before requesting a hold action.")
        self._observe()
        action = self._state.copy()
        action[-3:] = 0
        return action

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Send a measured-time control tick and return the next observation."""
        started = time.monotonic()
        action = np.array(action, dtype=np.float32, copy=True)
        if action.shape != (25,):
            raise ValueError(f"Expected SpiritAI action (25,), got {action.shape}.")
        if self._state is None:
            raise RuntimeError("Expected reset() before SpiritAI step().")
        if not np.isfinite(action).all():
            raise ValueError("Expected finite SpiritAI Cartesian targets.")
        from scipy.spatial.transform import Rotation

        cfg = self.config
        predicted_action = action.copy()
        measured_state = self._state.copy()
        for start, translation, rotation in (
            (0, cfg.max_arm_translation_m_s, cfg.max_arm_rotation_rad_s),
            (8, cfg.max_arm_translation_m_s, cfg.max_arm_rotation_rad_s),
            (16, cfg.max_torso_translation_m_s, cfg.max_torso_rotation_rad_s),
        ):
            part = slice(start, start + 3)
            delta = action[part] - self._state[part]
            distance = np.linalg.norm(delta)
            scale = min(1.0, translation / cfg.source_hz / max(distance, 1e-12))
            action[part] = self._state[part] + delta * scale
            part = slice(start + 3, start + 6)
            current = Rotation.from_rotvec(self._state[part])
            delta_rotation = (
                current.inv() * Rotation.from_rotvec(action[part])
            ).as_rotvec()
            angle = np.linalg.norm(delta_rotation)
            scale = min(1.0, rotation / cfg.source_hz / max(angle, 1e-12))
            action[part] = (
                current * Rotation.from_rotvec(delta_rotation * scale)
            ).as_rotvec()
        for index in (7, 15):
            limit = cfg.max_gripper_velocity_s / cfg.source_hz
            action[index] = np.clip(
                action[index], self._state[index] - limit, self._state[index] + limit
            )
            if cfg.gripper_min is not None:
                if (
                    cfg.execution_enabled
                    and not cfg.gripper_min - 1e-6
                    <= self._state[index]
                    <= cfg.gripper_max + 1e-6
                ):
                    raise ValueError(
                        "Expected measured gripper state within configured command bounds."
                    )
                action[index] = np.clip(action[index], cfg.gripper_min, cfg.gripper_max)
        action[-3:] = np.clip(action[-3:], -cfg.max_base_speed, cfg.max_base_speed)
        if self.config.zero_base_actions:
            action[-3:] = 0
        try:
            if cfg.execution_enabled:
                self._enable_following()
                self.client.send_actions(action[None], self.config.source_hz)
                self.client.wait_idle()
            observation = self._observe()
        except BaseException:
            self.close()
            raise
        remaining = 1 / self.config.source_hz - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
        return (
            observation,
            0.0,
            False,
            False,
            {
                "spiritai_control": {
                    "execution_enabled": cfg.execution_enabled,
                    "predicted_action": predicted_action,
                    "limited_action": action,
                    "measured_before": measured_state,
                    "measured_after": self._state.copy(),
                    "tick_seconds": time.monotonic() - started,
                }
            },
        )

    def close(self) -> None:
        """Release following mode acquired by this env and close the client."""
        try:
            try:
                if (
                    self.config.execution_enabled
                    and self.client.connected
                    and self.client.metadata.get("execution_capabilities", {}).get(
                        "cancel_actions"
                    )
                ):
                    self.client.cancel_actions()
                elif self._following_enabled and self.client.connected:
                    self.client.wait_idle()
            finally:
                if self._following_enabled and self.client.connected:
                    response = self.client.request(
                        {"type": "set_external_following", "enabled": False}
                    )
                    if (
                        not response.get("accepted")
                        or response.get("enabled") is not False
                    ):
                        raise RuntimeError(
                            f"Expected disabled external following, got {response!r}."
                        )
        finally:
            self._following_enabled = False
            self.client.close()
