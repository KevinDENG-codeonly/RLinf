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

"""Check the native SpiritAI bridge against loopback and fake robot hardware."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from robot_mocks.spiritai_container import SpiritaiContainer

from rlinf.robotics.spiritai import CAMERA_KEYS, SpiritaiClient
from rlinf.utils.logging import get_logger
from toolkits.realworld_check.check_spiritai import check_spiritai


def check_protocol(container: SpiritaiContainer) -> None:
    """Exercise actual server interpolation, completion, failure and cancellation."""
    client = SpiritaiClient(container.url)
    try:
        client.connect()
        client.validate_execution_capabilities()
        client.send_actions(np.zeros((1, 25), dtype=np.float32), 30)
        client.wait_idle()
        assert len(container.sdk.rows) == 1
        container.sdk.delay_s = 0.08
        client.send_actions(np.zeros((1, 25), dtype=np.float32), 30)
        time.sleep(0.02)
        assert client.request({"type": "get_status"})["busy"]
        client.wait_idle()
        container.sdk.delay_s = 0
        start = len(container.sdk.rows)
        client.send_actions(np.zeros((50, 25), dtype=np.float32), 30)
        time.sleep(0.05)
        client.cancel_actions()
        count = len(container.sdk.rows)
        time.sleep(0.05)
        assert len(container.sdk.rows) == count
        assert 0 < count - start < 197
        container.sdk.fail = True
        client.send_actions(np.zeros((1, 25), dtype=np.float32), 30)
        try:
            client.wait_idle()
        except RuntimeError as exc:
            assert "injected SDK" in str(exc)
        else:
            raise AssertionError("Expected observable post-ACK SDK failure.")
        client.cancel_actions()
    finally:
        client.close()


def main() -> None:
    """Run CPU protocol checks or one selected policy through native Ray workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--policy-kind", choices=("flow_only", "explicit_subtask"))
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sample-cache", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--gpu", type=int, default=0, help="One RLinf hardware rank for the policy."
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute only on this loopback fake SDK."
    )
    parser.add_argument("--inject-delay-ms", type=int, default=0)
    parser.add_argument("--bootstrap-timeout", type=float, default=30)
    parser.add_argument("--interrupt-after-steps", type=int, default=0)
    parser.add_argument("--signal-process-group", action="store_true")
    parser.add_argument("--sdk-fail-after-steps", type=int, default=0)
    parser.add_argument("--disable-guidance", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.protocol_only:
        with SpiritaiContainer(args.server_root) as container:
            check_protocol(container)
        (args.output / "summary.json").write_text(
            json.dumps(
                {"protocol_checks_passed": True, "real_robot_connected": False},
                indent=2,
            )
            + "\n"
        )
        return
    if not all((args.policy_kind, args.ckpt, args.bundle, args.sample_cache)):
        parser.error(
            "Policy checks require --policy-kind, --ckpt, --bundle and --sample-cache."
        )
    if args.steps <= 0 or args.steps % 50:
        parser.error("--steps must be positive and divisible by horizon 50.")
    repo = Path(__file__).resolve().parents[3]
    state = np.load(args.sample_cache / "states.npy", mmap_mode="r")[0].copy()
    pixels = np.load(args.sample_cache / "images.npy", mmap_mode="r")[0]
    images = {
        name: np.array(pixels[index, ::2, ::2], copy=True)
        for index, name in enumerate(CAMERA_KEYS)
    }
    name = "full" if args.policy_kind == "flow_only" else "subtask"
    profile = f"realworld_spiritai_eval_pi05_{name}_deploy"
    trace_path = args.output / f"spiritai_pi05_{name}_deploy/rtc-env-0.jsonl"
    with SpiritaiContainer(args.server_root, state=state, images=images) as container:
        report = check_spiritai(container.url, require_execution=True)
        assert report["actions_sent"] == 0 and not container.sdk.rows
        command = [
            sys.executable,
            str(repo / "evaluations/eval_embodied_agent.py"),
            "--config-path",
            "realworld",
            "--config-name",
            profile,
            f"cluster.component_placement.rollout={args.gpu}",
            f"rollout.model.model_path={args.ckpt.resolve()}",
            f"rollout.model.openpi.deployment_metadata_path={args.bundle.resolve()}/openpi_deployment.json",
            f"rollout.model.openpi_data.norm_stats_path={args.bundle.resolve()}/norm_stats.json",
            f"env.eval.override_cfg.robot_url={container.url}",
            f"env.eval.override_cfg.execution_enabled={str(args.execute).lower()}",
            f"env.eval.max_steps_per_rollout_epoch={args.steps}",
            f"env.eval.max_episode_steps={args.steps}",
            f"runner.rtc.inject_delay_ms={args.inject_delay_ms}",
            f"runner.rtc.bootstrap_timeout_s={args.bootstrap_timeout}",
            f"rollout.model.openpi.rtc_enabled={str(not args.disable_guidance).lower()}",
            f"runner.logger.log_path={args.output.resolve()}",
            "runner.logger.logger_backends=[]",
        ]
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "RLINF_ROBOT_MOCKS": "1",
            "HYDRA_FULL_ERROR": "1",
            "EMBODIED_PATH": str(repo / "examples/embodiment"),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(repo),
                    str(repo / "tests"),
                    str(repo / "tests/unit_tests"),
                    str(repo / "tests/robot_mocks"),
                ]
            ),
        }
        (args.output / "command.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
                },
                indent=2,
            )
            + "\n"
        )
        interrupted = False
        with (args.output / "driver.log").open("w") as log:
            child = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + args.steps / 30 + 240
            try:
                while child.poll() is None:
                    if (
                        (args.interrupt_after_steps or args.sdk_fail_after_steps)
                        and not interrupted
                        and trace_path.exists()
                    ):
                        steps = sum(
                            json.loads(line)["event"] == "step"
                            for line in trace_path.read_text().splitlines()
                            if line.endswith("}")
                        )
                        threshold = (
                            args.interrupt_after_steps or args.sdk_fail_after_steps
                        )
                        if steps >= threshold:
                            if args.interrupt_after_steps:
                                if args.signal_process_group:
                                    os.killpg(child.pid, signal.SIGINT)
                                else:
                                    child.send_signal(signal.SIGINT)
                            else:
                                container.sdk.fail = True
                            interrupted = True
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Native bridge test exceeded its deadline.")
                    time.sleep(0.05)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        assert trace_path.exists(), (
            "Native evaluation did not reach the RTC env worker."
        )
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        aborts = [event for event in events if event["event"] == "abort"]
        assert events[-1]["event"] == "closed"
        assert not container.sdk.following
        assert bool(container.sdk.rows) == (
            args.execute and args.bootstrap_timeout >= 1
        )
        responses = [event for event in events if event["event"] == "response"]
        if args.bootstrap_timeout < 1:
            assert len(aborts) == 1 and "RTC warmup request" in aborts[0]["error"]
            assert not any(event["event"] in ("step", "hold") for event in events)
        elif args.inject_delay_ms:
            assert len(aborts) == 1 and "RTC replan request" in aborts[0]["error"]
            assert sum(event["event"] == "hold" for event in events) == 2
        elif args.interrupt_after_steps:
            assert interrupted and len(aborts) == 1 and "SIGINT" in aborts[0]["error"]
        elif args.sdk_fail_after_steps:
            assert (
                interrupted
                and len(aborts) == 1
                and "injected SDK dispatch failure" in aborts[0]["error"]
            )
        else:
            assert not aborts and child.returncode == 0
            assert sum(event["event"] == "step" for event in events) == args.steps
        if aborts:
            assert child.returncode != 0
        if args.policy_kind == "flow_only":
            assert not any("subtasks" in event for event in responses)
        else:
            assert any(
                event.get("subtasks") and all(event.get("subtask_eos", [True]))
                for event in responses
            )
        if args.disable_guidance:
            assert not any(event["guidance_applied"] for event in responses)
        replan_times = [
            event["inference_seconds"] for event in responses if event["chunk_id"] >= 3
        ]
        summary = {
            "policy_kind": args.policy_kind,
            "returncode": child.returncode,
            "sdk_action_rows": len(container.sdk.rows),
            "steps": sum(event["event"] == "step" for event in events),
            "aborts": aborts,
            "closed": True,
            "external_following_after_exit": container.sdk.following,
            "real_robot_connected": False,
            "inference_requests": len(responses),
            "replan_seconds_percentiles": {
                str(q): float(np.percentile(replan_times, q)) for q in (50, 95, 99)
            }
            if replan_times
            else {},
            "checks_passed": True,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        get_logger().info("SpiritAI bridge check: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
