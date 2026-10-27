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

"""Reusable OpenPI data transforms for SpiritAI Cartesian control."""

import dataclasses
import pathlib

from openpi.models.model import BaseModelConfig
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from openpi.transforms import Group

from rlinf.models.embodiment.openpi.policies.spiritai_policy import (
    ACTION_KEYS,
    SpiritaiCartesianInputs,
    SpiritaiCartesianOutputs,
)


@dataclasses.dataclass(frozen=True)
class LeRobotSpiritaiCartesianDataConfig(DataConfigFactory):
    """Configure SpiritAI data without embedding a task prompt or dataset path."""

    zero_base_action_targets: bool = False

    def create(
        self, assets_dirs: pathlib.Path, model_config: BaseModelConfig
    ) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=Group(
                inputs=[SpiritaiCartesianInputs(self.zero_base_action_targets)],
                outputs=[SpiritaiCartesianOutputs()],
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=ACTION_KEYS,
        )
