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

"""SpiritAI Cartesian demonstrations and real-robot observations."""

import dataclasses

import numpy as np
from openpi import transforms

from rlinf.robotics.spiritai import (
    ACTION_DIM,
    ACTION_KEYS,
    CAMERA_KEYS,
    STATE_KEYS,
    pack_cartesian_fields,
)


def _image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a three-dimensional RGB image, got {image.shape}.")
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with three channels, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class SpiritaiCartesianInputs(transforms.DataTransformFn):
    """Map LeRobot fields or packed env observations to OpenPI inputs."""

    zero_base_action_targets: bool = False

    def __call__(self, data: dict) -> dict:
        if "observation/state" in data:
            state = np.asarray(data["observation/state"], dtype=np.float32)
            views = data["observation/extra_view_image"]
            cameras = (data["observation/image"], views[0], views[1])
        else:
            state = pack_cartesian_fields(data, STATE_KEYS)
            cameras = tuple(data[key] for key in CAMERA_KEYS)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError(
                f"Expected finite SpiritAI state (25,), got {state.shape}."
            )
        images = {
            key: _image(value)
            for key, value in zip(
                ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
                cameras,
                strict=True,
            )
        }
        result = {
            "state": state,
            "image": images,
            "image_mask": dict.fromkeys(images, np.True_),
        }
        if ACTION_KEYS[0] in data:
            actions = pack_cartesian_fields(data, ACTION_KEYS)
        else:
            actions = data.get("actions")
        if actions is not None:
            actions = np.array(actions, dtype=np.float32, copy=True)
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"Expected SpiritAI actions (H, 25), got {actions.shape}."
                )
            if not np.isfinite(actions).all():
                raise ValueError("Expected finite SpiritAI action targets.")
            if self.zero_base_action_targets:
                actions[..., -3:] = 0
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class SpiritaiCartesianOutputs(transforms.DataTransformFn):
    """Return the unnormalized 25-D Cartesian policy action."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"Expected model actions (H, D>=25), got {actions.shape}.")
        if not np.isfinite(actions).all():
            raise ValueError("Expected finite SpiritAI model actions.")
        return {"actions": actions[:, :ACTION_DIM]}
