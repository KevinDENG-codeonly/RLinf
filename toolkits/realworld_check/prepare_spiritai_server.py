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

"""Prepare a verified robot_server compatibility copy for review and deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from rlinf.utils.logging import get_logger


def prepare_server(source_root: str | Path, output: str | Path) -> Path:
    """Patch a new copy of the supported server version, preserving the source."""
    source = Path(source_root).expanduser().resolve() / "robot_server"
    destination = Path(output).expanduser().resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise FileExistsError(
            f"Expected a new output outside {source}, got {destination}."
        )
    artifacts = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/pi05-explicit-subtask-ce-lora/artifacts"
    )
    manifest = json.loads((artifacts / "robot_server_compat_manifest.json").read_text())
    for name, identity in manifest["files"].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != identity["source_sha256"]:
            raise ValueError(
                f"Expected supported unpatched {name} SHA256, got {actual}; review this server version first."
            )
    try:
        shutil.copytree(
            source,
            destination / "robot_server",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        subprocess.run(
            [
                "patch",
                "--batch",
                "--fuzz=0",
                "-p1",
                "-i",
                str(artifacts / "robot_server_compat.patch"),
            ],
            cwd=destination,
            check=True,
            capture_output=True,
            text=True,
        )
        for name, identity in manifest["files"].items():
            actual = hashlib.sha256(
                (destination / "robot_server" / name).read_bytes()
            ).hexdigest()
            if actual != identity["patched_sha256"]:
                raise ValueError(
                    f"Expected reviewed patched {name} SHA256, got {actual}."
                )
        (destination / "compatibility.json").write_text(
            json.dumps(
                {
                    "source_root": str(source.parent),
                    "files": manifest["files"],
                    "applied_to_robot": False,
                },
                indent=2,
            )
            + "\n"
        )
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    get_logger().info("Prepared reviewed server copy at %s", destination)
    return destination


def main() -> None:
    """Create a compatibility copy without starting a server or importing SDKs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_server(args.source_root, args.output)


if __name__ == "__main__":
    main()
