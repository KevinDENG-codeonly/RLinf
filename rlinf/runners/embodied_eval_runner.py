# Copyright 2025 The RLinf Authors.
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

import signal
import time
import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        # Channel direction convention (names follow the receiver, not the sender):
        #   rollout_channel: env -> rollout  (env sends obs/RTC requests, rollout receives)
        #   env_channel:     rollout -> env  (rollout sends actions/RTC responses, env receives)
        rtc = self.cfg.runner.get("rtc", {})
        bounded_rtc = (
            bool(rtc.get("enabled", False))
            and float(rtc.get("request_timeout_s", 0)) > 0
        )
        if bounded_rtc:
            env_results, rollout_results = self._evaluate_bounded_rtc()
        else:
            env_handle: Handle = self.env.evaluate(
                input_channel=self.env_channel,
                rollout_channel=self.rollout_channel,
            )
            rollout_handle: Handle = self.rollout.evaluate(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            )
            env_results = env_handle.wait()
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            if not bounded_rtc:
                timeout = float(rtc.get("shutdown_timeout_s", 0))
                if timeout > 0:
                    deadline = time.monotonic() + timeout
                    while not rollout_handle.done():
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "RTC rollout did not stop after env completion."
                            )
                        time.sleep(0.01)
                rollout_results = rollout_handle.wait()
            rollout_metrics_list = [
                results for results in rollout_results if results is not None
            ]
            rollout_metrics = compute_evaluate_metrics(rollout_metrics_list)
            rollout_metrics.pop("num_trajectories", None)
        else:
            rollout_metrics = {}

        env_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(env_metrics_list)
        eval_metrics.update(rollout_metrics)
        return eval_metrics

    def _evaluate_bounded_rtc(self):
        """Let the env close its connection before a signalled driver exits."""
        control = Channel.create("RTCControl")
        timeout = float(self.cfg.runner.rtc.shutdown_timeout_s)
        if timeout <= 0:
            raise ValueError(
                f"Expected positive RTC shutdown_timeout_s, got {timeout}."
            )
        stop_reason = None
        stop_sent_at = None
        env_handle = rollout_handle = None

        def request_stop(signum, frame):
            nonlocal stop_reason
            stop_reason = signal.Signals(signum).name

        saved = {
            sig: signal.getsignal(sig)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)
        }
        for sig in saved:
            signal.signal(sig, request_stop)
        try:
            env_handle = self.env.evaluate(
                input_channel=self.env_channel,
                rollout_channel=self.rollout_channel,
                control_channel=control,
            )
            rollout_handle = self.rollout.evaluate(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            )
            while not env_handle.done():
                if rollout_handle.done() and stop_reason is None:
                    stop_reason = "rollout_finished_before_env"
                if stop_reason is not None and stop_sent_at is None:
                    control.put(stop_reason, key=0)
                    stop_sent_at = time.monotonic()
                if (
                    stop_sent_at is not None
                    and time.monotonic() - stop_sent_at >= timeout
                ):
                    raise TimeoutError(
                        "RTC env did not acknowledge shutdown before the deadline."
                    )
                time.sleep(0.01)
            env_results = env_handle.wait()
            deadline = time.monotonic() + timeout
            while not rollout_handle.done():
                if time.monotonic() >= deadline:
                    raise TimeoutError("RTC rollout did not stop after env cleanup.")
                time.sleep(0.01)
            return env_results, rollout_handle.wait()
        except BaseException:
            if env_handle is not None and not env_handle.done():
                control.put("driver_failure", key=0)
                deadline = time.monotonic() + timeout
                while not env_handle.done() and time.monotonic() < deadline:
                    time.sleep(0.01)
            raise
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)

    def run(self):
        start_time = time.time()
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)
        print_metrics_table(
            step=0,
            total_steps=1,
            start_time=start_time,
            metrics=eval_metrics,
            log_path=self.metric_logger.log_path,
        )

        self.metric_logger.finish()
        if float(eval_metrics.get("eval/rtc_aborted", 0.0)) > 0:
            raise RuntimeError(
                "RTC evaluation aborted; inspect rtc-env-<rank>.jsonl for the failed "
                "request, hold attempts and connection cleanup."
            )
