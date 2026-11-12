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

"""Real-Time Control (RTC) rollout worker for real-world OpenPI evaluation.

This subclass overrides :meth:`evaluate` to serve RTC bootstrap/replan
requests from the env worker, applying flow-matching guidance so each new
action chunk stays consistent with the unexecuted tail of the previous one.
"""

import time

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig
from tqdm import tqdm

from rlinf.data.schema.embodied_types import RTCActionResponse, RTCRequest
from rlinf.envs import SupportedEnvType
from rlinf.scheduler import Channel, Worker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class RTCMultiStepRolloutWorker(MultiStepRolloutWorker):
    """Rollout worker that serves RTC requests during evaluation.

    Channel convention (same as the base class): ``input_channel`` carries
    data received from the env worker (RTC requests) and ``output_channel``
    carries data sent to the env worker (RTC action responses).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._rtc_eval_model_actions: torch.Tensor | None = None
        self._subtask_state = None
        self._rtc_session_id = ""
        rtc_cfg = self.cfg.runner.get("rtc", {})
        self.inject_delay_ms = float(rtc_cfg.get("inject_delay_ms", 0.0))
        self.fixed_delay_steps = int(rtc_cfg.get("fixed_delay_steps", 0))
        self.eval_chunk_pause_seconds = float(rtc_cfg.get("chunk_pause_seconds", 0.0))
        self.require_subtask_eos = bool(rtc_cfg.get("require_subtask_eos", False))

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        """Serve RTC requests until the env worker sends an explicit stop."""
        if self.enable_offload:
            self.reload_model()

        self._rtc_eval_model_actions = None
        if self._subtask_state is not None:
            self._subtask_state.reset()

        pbar = tqdm(desc="Evaluating RTC Rollout", disable=(self._rank != 0))
        try:
            while True:
                rtc_request: RTCRequest = await self.recv_rtc_request(input_channel)
                if rtc_request.request_type == "stop":
                    break

                try:
                    rtc_response = self._predict_rtc(rtc_request)
                except Exception as exc:
                    if not rtc_request.session_id:
                        raise
                    self.log_error(f"RTC inference failed: {exc}")
                    rtc_response = RTCActionResponse(
                        chunk_id=rtc_request.chunk_id,
                        session_id=rtc_request.session_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                self.send_rtc_response(output_channel, rtc_response)
                pbar.update(1)
        finally:
            pbar.close()

        if self.enable_offload:
            self.offload_model()

    @Worker.timer("predict_rtc")
    def _predict_rtc(self, rtc_request: RTCRequest) -> RTCActionResponse:
        """Run one OpenPI inference for a bootstrap or replanning RTC request."""
        start_time = time.perf_counter()
        if rtc_request.request_type in ("bootstrap", "warmup"):
            self._rtc_eval_model_actions = None
            if self._subtask_state is not None:
                self._subtask_state.reset()
            self._rtc_session_id = rtc_request.session_id
        elif rtc_request.session_id != self._rtc_session_id:
            raise RuntimeError(
                "Expected bootstrap before replanning a new RTC session."
            )
        rtc_context = None
        guidance_applied = False
        hierarchy = getattr(self.hf_model, "hierarchical_cfg", None)
        planning_kwargs = {}
        if hierarchy is not None and hierarchy.enabled:
            from rlinf.models.embodiment.openpi.hierarchical import SubtaskPlanningState

            if self._subtask_state is None:
                self._subtask_state = SubtaskPlanningState()
            planning_kwargs["subtask_state"] = self._subtask_state
            planning_kwargs["require_subtask_eos"] = self.require_subtask_eos
        if (
            rtc_request.request_type == "replan"
            and self._rtc_eval_model_actions is not None
        ):
            from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
                RTCGuidanceContext,
            )

            rtc_context = RTCGuidanceContext(
                prev_model_actions=self._rtc_eval_model_actions,
                executed_horizon=rtc_request.executed_horizon,
                delay_steps=rtc_request.predicted_delay_steps,
            )
            guidance_applied = True

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=rtc_request.obs,
                mode="eval",
                rtc_context=rtc_context,
                **planning_kwargs,
            )
            if rtc_request.request_type == "warmup" and getattr(
                self.hf_model, "rtc_enabled", False
            ):
                from rlinf.models.embodiment.openpi.sampling.rtc_guidance import (
                    RTCGuidanceContext,
                )

                warmup_context = RTCGuidanceContext(
                    prev_model_actions=result["model_actions"],
                    executed_horizon=0,
                    delay_steps=rtc_request.predicted_delay_steps,
                )
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=rtc_request.obs,
                    mode="eval",
                    rtc_context=warmup_context,
                    **planning_kwargs,
                )
        guidance_applied = result.get("rtc_guidance_applied", guidance_applied)
        if result.get("subtask_regenerated", False):
            self.log_info(
                f"Generated subtasks={result['subtasks']}, "
                f"planning_seconds={result['planning_seconds']:.3f}"
            )

        if rtc_request.request_type == "replan":
            if (
                SupportedEnvType(self.cfg.env.eval.env_type)
                is not SupportedEnvType.REAL
                and self.eval_chunk_pause_seconds > 0
            ):
                delay_steps = max(
                    self.fixed_delay_steps, rtc_request.predicted_delay_steps
                )
                target_time = (delay_steps - 1) * self.eval_chunk_pause_seconds
                elapsed = time.perf_counter() - start_time
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
            elif (
                SupportedEnvType(self.cfg.env.eval.env_type) is SupportedEnvType.REAL
                and self.inject_delay_ms > 0
            ):
                target_time = self.inject_delay_ms / 1000.0
                elapsed = time.perf_counter() - start_time
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        model_actions = result.get("model_actions")
        if isinstance(model_actions, np.ndarray):
            model_actions = torch.from_numpy(model_actions)
        if model_actions is not None:
            self._rtc_eval_model_actions = model_actions.detach().cpu().contiguous()

        response = RTCActionResponse(
            actions=actions.detach().cpu().contiguous(),
            model_actions=self._rtc_eval_model_actions,
            chunk_id=rtc_request.chunk_id,
            guidance_applied=guidance_applied,
            session_id=rtc_request.session_id,
            diagnostics={
                key: result[key]
                for key in (
                    "subtasks",
                    "subtask_regenerated",
                    "subtask_changed",
                    "subtask_eos",
                    "planning_seconds",
                )
                if key in result
            },
        )
        response.diagnostics["inference_seconds"] = time.perf_counter() - start_time
        if rtc_request.request_type == "warmup":
            self._rtc_eval_model_actions = None
            if self._subtask_state is not None:
                self._subtask_state.reset()
        return response

    async def recv_rtc_request(self, input_channel: Channel) -> RTCRequest:
        """Receive the single real-world RTC request mapped to this rollout rank."""
        return await self.recv_from(
            group_name=self.cfg.env.group_name,
            channel=input_channel,
            tag="eval_rtc",
            route_key=0,
            async_op=True,
            batch_size=self.total_num_eval_envs,
            merge_fn=lambda items: items[0],
            infer_batch_size_fn=lambda data: 1,
        ).async_wait()

    def send_rtc_response(
        self, output_channel: Channel, rtc_response: RTCActionResponse
    ) -> None:
        """Send the RTC action response back to the matching env rank."""
        self.send_to(
            group_name=self.cfg.env.group_name,
            channel=output_channel,
            data=rtc_response,
            tag="eval_rtc",
            route_key=0,
            batch_size=self.total_num_eval_envs,
            split_fn=lambda data, sizes: [data],
        )
