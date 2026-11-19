# Full Flow Inference Parameters

Adjust run length, execution, RTC, sampling, and limits by purpose, and check which configuration layer owns the setting. This page describes current behavior without changing project YAML. [Runbook](README.en.md) · [中文](parameters.md)

## Configuration Sources

`key=value` is a Hydra startup override, not a model observation or mandatory extra API input. The primary loads its defaults, applies `_self_`, then CLI values override matching keys. Paths below are relative to the repository root.

| File | Settings it owns |
|---|---|
| [`realworld_spiritai_eval_pi05_full_deploy.yaml`](../../../evaluations/realworld/realworld_spiritai_eval_pi05_full_deploy.yaml) | Full Flow primary, mode, and persistent experiment overrides |
| [`_spiritai_pi05_deploy.yaml`](../../../evaluations/realworld/_spiritai_pi05_deploy.yaml) | Shared observer default, deadlines, three images, BF16, sampling, and limit overrides |
| [`_spiritai_pi05_rtc.yaml`](../../../evaluations/realworld/_spiritai_pi05_rtc.yaml) | Native RTC/worker topology, replanning, and W&B default |
| [`realworld_spiritai_cartesian.yaml`](../../../examples/embodiment/config/env/realworld_spiritai_cartesian.yaml) | Env template, default lengths, endpoint, cameras, and following |
| [`SpiritaiCartesianConfig`](../../../rlinf/envs/real/spiritai/cartesian.py) | Class defaults for translation, rotation, and gripper velocity when YAML does not override them |

Prefer CLI overrides for individual runs; startup logs record the resolved configuration. Persist Full Flow changes in its primary. Changing shared settings affects inheriting profiles. A primary override continues to win over later edits to a lower template. Current commands explicitly provide checkpoint and bundle paths because the primary still contains placeholder paths.

## Why Three Common Overrides Appear Together

Distinguish the worker's outer step budget from the environment's episode boundary. In bounded RTC, each `_evaluate_rtc_action()` sends one target row/calls one environment step rather than executing a complete 50-row chunk.

| Parameter | Default and source | Effect |
|---|---|---|
| `env.eval.max_steps_per_rollout_epoch` | `3000`, env template | Bounds the `RTCEnvWorker._evaluate_bounded` outer control loop per evaluation round |
| `env.eval.max_episode_steps` | `3000`, env template | `RealEnv.step` compares elapsed steps with this limit to produce timeout truncation/done |
| `env.eval.override_cfg.execution_enabled` | `false`, shared deployment settings | False predicts/reads only; true dispatches and manages following as configured |

The worker breaks on done or ends when its loop budget runs out. With current single-env `rollout_epoch=1`, `auto_reset=false`, and no earlier termination/error, the smaller limit normally determines the tick count.

| Outer budget | Episode limit | Normal boundary |
|---:|---:|---|
| 1500 | 1500 | 1500 ticks, aligned budget and truncation |
| 1500 | 3000 | 1500 ticks, loop budget; env may not be done |
| 3000 | 1500 | 1500 ticks, episode truncation breaks the loop |
| 1500 | 500 | 500 ticks, episode boundary first |

Equality is an operating convention, not an API requirement. The generic validator requires **`max_steps_per_rollout_epoch` divisible by `num_action_chunks=50`**. Use lengths such as 50, 300, 1500, or 3000. `max_episode_steps` has no corresponding divisibility check there; matching positive values keeps boundaries clear.

`execution_enabled=false` still runs predictions and counts preview control steps. With true, dispatch starts after warmup/bootstrap without a keyboard confirmation and without moving to an initial preset. The env class itself defaults execution to true, but this deployment YAML explicitly overrides it to false; use the specified deployment primary for the defaults described here.

## Omitting Repeated CLI Arguments

Omit all three when defaults `3000 / 3000 / false` suit the run. To persist a 1500-tick preview, merge this example into the Full Flow primary. It is documentation only; this update does not edit the config:

```yaml
env:
  eval:
    max_steps_per_rollout_epoch: 1500
    max_episode_steps: 1500
    override_cfg:
      execution_enabled: false
```

Then append only `env.eval.override_cfg.execution_enabled=true` when motion is intended. Persisting true in YAML is also possible, but every launch of that profile would execute after warmup. Current defaults remain false to distinguish preview from motion commands.

Persist `rollout.model.model_path`, `rollout.model.openpi.deployment_metadata_path`, and `rollout.model.openpi_data.norm_stats_path` in the primary to omit path overrides too. Assets must exist and match the same weights. Images, state, and `task_description` remain the model inputs; these operational settings are not prompt text.

## Run Length and Rates

Estimate action duration using measured control rate, then choose a nearby valid step count. `source_hz` defines the nominal action timeline and limit calculation, not guaranteed closed-loop throughput.

| Parameter | Default | Effect and boundary |
|---|---:|---|
| `env.eval.rollout_epoch` | 1 | Evaluation rounds; increasing repeats reset/bootstrap rather than extending one continuous trajectory |
| `env.eval.auto_reset` | false | The current worker owns episode boundaries; retain false for this robot profile |
| `env.eval.override_cfg.source_hz` | 30 | Time basis for action indices, overlap delay, and per-tick limits; match training rather than changing it to extend duration |
| `env.eval.override_cfg.robot_url` | `ws://172.16.0.30:8766` | Robot endpoint; server bind address `0.0.0.0` is not a client destination |
| `cluster.component_placement.rollout` | Inherits 0; local command uses 1 | Select one policy GPU hardware rank, not two GPUs |
| `cluster.component_placement.env` | 0 | Env node placement, on the workstation CPU side; not another GPU model |

Action duration is approximately `N / f_measured`. The 1500-tick hardware run on 2026-11-19 spanned 58.97 seconds between first/last steps at 25.42 Hz; nominal 30 Hz would imply about 50 seconds. About `60 × 25 ≈ 1500` ticks targets a minute. Initialization, 13 GB weight hashing/loading, and compilation warmup are additional. Remeasure when throughput changes. There is no documented exact wall-clock 60-second setting here; `max_episode_steps=60` means steps, not seconds.

## RTC and Flow Sampling

Control asynchronous scheduling and sampler guidance separately to vary the intended mechanism. Full Flow uses the verified `approx` sampler; deployment does not add training objectives to the checkpoint.

| Parameter | Current default | Effect and adjustment |
|---|---|---|
| `runner.rtc.enabled` | true | Select native asynchronous RTC env/rollout workers; retain for normal deployment |
| `rollout.model.openpi.rtc_enabled` | true | Apply guidance when a previous chunk exists; false can retain the same asynchronous controller for an isolated comparison |
| `rollout.model.openpi.rtc_guidance_mode` | `approx` | Supported deployment guidance; distinct from old OpenPI policy-server `training_time_v1` metadata |
| `rollout.model.openpi.rtc_guidance_clip` | 5.0 | Cap the Flow-time-dependent guidance scale, not Cartesian velocity |
| `runner.rtc.min_exec_horizon` | 10 | Request replacement when the action-time index reaches at least ten and no request is pending; query rate also depends on inference/RPC latency |
| `runner.rtc.initial_delay_steps` | 15 | Initial overlap delay estimate, 0.5 seconds at 30 Hz; not a startup sleep |
| `runner.rtc.delay_buffer_size` | 8 | Window of recent replan delays, using its maximum; the initial estimate ages out |
| `runner.rtc.request_timeout_s` | 1.2 | Local wall-clock replan budget; rejects late actions but does not hard-preempt every blocking RPC |
| `runner.rtc.bootstrap_timeout_s` | 30.0 | Budget per warmup/bootstrap request, excluding earlier weight loading |
| `runner.rtc.warmup_requests` | 2 | Initial requests covering plain Flow and RTC paths, followed by reset/bootstrap |
| `runner.rtc.shutdown_timeout_s` | 15.0 | Driver budget for worker cleanup |
| `runner.rtc.stop_hold_steps` | 2 | At most two measured-pose hold attempts after a completed control tick and subsequent failure; not extra normal-exit moves or guaranteed successful holds |
| `env.eval.override_cfg.timeout_s` | 2.0 | Individual robot connection/RPC and idle-wait deadlines, separate from model replanning |
| `runner.rtc.fixed_delay_steps` / `runner.rtc.chunk_pause_seconds` | 0 / 0.0 | Simulation settings that must remain zero on this real-world path |
| `runner.rtc.inject_delay_ms` | 0 | Artificial fault injection; retain zero during normal robot runs |
| `rollout.model.num_steps` / `rollout.model.openpi.num_steps` | 10 / 10 | Euler iterations; the factory prefers the latter. Override both together. Fewer forward passes reduce work but require action-quality validation |
| `rollout.model.openpi.torch_compile` | true | Existing local Torch compilation; affects first-call warmup and steady overhead, not training or a whole-model build |
| `runner.rtc.require_subtask_eos` | true, shared setting | Full Flow hierarchy=false does not generate Subtasks, so this field needs no adjustment |

Horizon 50 covers about 1.67 seconds on the 30 Hz action timeline. RTC requests replacements before all fifty rows finish. Replies skip elapsed prefixes using local monotonic time. Exhaustion aborts rather than repeating the last target indefinitely. A replacement must meet the request budget and retain usable action time.

## Execution and Limits

Execution determines whether a command is submitted; limits constrain its target first. Velocity defaults below come from `SpiritaiCartesianConfig`, with shared deployment overrides for gripper/base settings.

| Parameter (all prefixed `env.eval.override_cfg.`) | Deployment default | Effect |
|---|---:|---|
| `enable_external_following` | true | Request following before the first valid action in execution mode; preview leaves mode unchanged |
| `defer_external_following` | true | Defer acquiring following until after warmup/initial inference |
| `require_execution_capabilities` | true | Check this adapter's protocol before dispatch; flags or a bypass are not replacements for behavior |
| `min_cartesian_dim` | 20 | Retain torso, rejecting 14D-only endpoints; 23D also carries zero base |
| `max_arm_translation_m_s` | 0.08 | Per-tick arm target displacement around measurement, m/s |
| `max_arm_rotation_rad_s` | 0.35 | Arm SO(3) angular speed cap, rad/s |
| `max_torso_translation_m_s` | 0.04 | Torso translation cap, m/s |
| `max_torso_rotation_rad_s` | 0.2 | Torso SO(3) angular cap, rad/s |
| `max_gripper_velocity_s` | 0.8 | Gripper command-unit change per second; not an unverified metres/second specification |
| `gripper_min` / `gripper_max` | 0.0 / 0.1 | Target range matching this checkpoint's approximately 0–0.1 action statistics; do not blindly substitute old reset=1.0 |
| `zero_base_actions` / `max_base_speed` | true / 0.0 | Keep base commands zero in this folding experiment |

Limits use measured state and `velocity / source_hz` per tick; measured throughput does not automatically replace `source_hz`. Rotation vectors use SO(3) differences, not componentwise Euler clipping. Software cancellation stops subsequent SDK submissions without withdrawing targets already accepted by the controller. Following mode alone does not mean commands have been sent.

## Model Contract, Instruction and Logs

These fields determine training-data interpretation or asset identity. Keep them consistent and use validation to detect mismatches; they are not duration or speed controls.

| Parameter | Current value / purpose |
|---|---|
| `rollout.model.openpi.policy_kind` / `rollout.model.openpi.hierarchical.enabled` | `flow_only` / false, overall-conditioned actions |
| `rollout.model.openpi.config_name` | `pi05_spiritai_cartesian`, input/output transforms |
| `rollout.model.num_action_chunks` / `rollout.model.openpi.action_chunk` | 50 / 50, horizon/output chunk rather than whole-run length |
| `rollout.model.action_dim` / `rollout.model.openpi.action_env_dim` / `rollout.model.openpi.model_action_dim` | 25 / 25 / 32, policy/state versus model-space dimensions |
| `rollout.model.openpi.num_images_in_input` | 3, high/left/right wrist, RGB 240×320 transformed to 224×224 |
| `rollout.model.openpi.discrete_state_input` / `rollout.model.openpi.max_token_len` | true / 256, training state-token and text contract |
| `rollout.model.precision` / `rollout.model.openpi.embed_dtype` | bf16 / bf16 with existing FP32 islands |
| `rollout.model.is_lora` | false, complete native weights |
| `rollout.model.openpi.validate_deployment` | true, validate mode, geometry, statistics, and complete weight SHA256 |
| `rollout.model.openpi_data.norm_stats_path` / `rollout.model.openpi_data.zero_base_action_targets` | Bundle statistics / true, training-compatible interpretation |
| `env.eval.override_cfg.image_height` / `env.eval.override_cfg.image_width` | 240 / 320, actual container image shape |
| `env.eval.override_cfg.task_description` | Overall instruction; do not append a current Subtask for Full Flow |
| `env.eval.keyboard_reward_wrapper` | null, no press-a keyboard start gate |
| `env.eval.runtime_setup` | false, external SDK container with local ROS processes preserved |
| `runner.logger.logger_backends` | YAML `[wandb]`; local commands override `[]`, retaining terminal/RTC traces |
| `runner.logger.project_name` / `runner.logger.experiment_name` | `rlinf` / `spiritai_pi05_full_deploy`, cloud naming and trace subdirectory |
| `runner.logger.log_path` | Timestamp directory generated by the launcher; normally use its default |

Quote the complete Hydra argument when the overall instruction contains spaces:

```bash
'env.eval.override_cfg.task_description="Fold the cardboard box and place it on the right side of the table."'
```

The shared model template also prints fields such as `noise_method`, `train_expert_only`, and `lora_rank`. With `task=eval / is_lora=false`, training/other-task fields should not be assumed to control this inference path.

This is an argument fragment to append to the launcher, not a standalone shell command. See the [runbook](README.en.md) for complete commands, W&B endpoint, and diagnosis.
