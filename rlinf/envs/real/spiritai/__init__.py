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

"""SpiritAI tasks using the robot's existing container."""

from rlinf.envs.real.registry import register_tasks
from rlinf.envs.real.spiritai.cartesian import SpiritaiCartesianEnv

TASKS = {"SpiritaiCartesianEnv-v1": SpiritaiCartesianEnv}
_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = ["SpiritaiCartesianEnv", "TASKS", *_ENTRY_POINTS]
