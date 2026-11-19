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

"""Check a SpiritAI endpoint without changing following or sending actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.robotics.spiritai import (
    CAMERA_KEYS,
    STATE_KEYS,
    SpiritaiClient,
    pack_cartesian_fields,
)
from rlinf.utils.logging import get_logger


def check_spiritai(
    url: str, *, timeout_s: float = 2.0, require_execution: bool = False
) -> dict[str, Any]:
    """Return endpoint metadata and measured shapes, always releasing the socket."""
    client = SpiritaiClient(url, timeout_s=timeout_s)
    try:
        client.connect()
        if client.command_dim < 20:
            raise ValueError(
                f"Expected whole-body Cartesian dimension >=20, got {client.command_dim}."
            )
        if require_execution:
            client.validate_execution_capabilities()
        status = client.request({"type": "get_status"})
        fields, images = client.observation()
        state = pack_cartesian_fields(fields, STATE_KEYS)
        if state.shape != (25,):
            raise ValueError(f"Expected measured state (25,), got {state.shape}.")
        shapes = {}
        for name in CAMERA_KEYS:
            pixels = np.asarray(images[name])
            if pixels.ndim != 3 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
                raise ValueError(
                    f"Expected RGB uint8 {name}, got {pixels.shape}/{pixels.dtype}."
                )
            shapes[name] = list(pixels.shape)
        return {
            "robot_url": url,
            "metadata": client.metadata,
            "status": status,
            "state": state.tolist(),
            "images": shapes,
            "actions_sent": 0,
            "following_changed": False,
        }
    finally:
        client.close()


def main() -> None:
    """Run a read-only check before starting the exclusive inference bridge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--require-execution",
        action="store_true",
        help="Check execution capabilities without enabling motion.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_spiritai(
        args.url, timeout_s=args.timeout_s, require_execution=args.require_execution
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    get_logger().info("SpiritAI preflight: %s", json.dumps(report))


if __name__ == "__main__":
    main()
