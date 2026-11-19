# SpiritAI inference bridge: implementation record

This is the 2026-11-18 software-stage snapshot, preserving its deployment state and commands. Use the current Full Flow [runbook](README.en.md) for operation.

Choose one Full Flow or Explicit Subtask LoRA policy through native RLinf evaluation, loading one model on one GPU per run. This record covers implementation, deployment assets, verification limits, and commands for this workstation. See the [approved proposal](research-proposal.en.md) for the design. [中文](implementation.md)

Implementation and verification date: 2026-11-18.

## Implementation and Ownership

The working branch is `feature/spiritai-unified-inference-bridge`, based on `f1d402ae`. The launch path remains `evaluations/run_eval.sh`, using `EmbodiedEvalRunner`, `RTCEnvWorker`, `RTCMultiStepRolloutWorker`, `Pi0Eval`, `SpiritaiCartesianEnv`, and `SpiritaiClient`. Only the env side connects to the robot container. Both policies share camera handling, decoding, limits, RTC scheduling, and transport, and run separately.

| Entry or file | Responsibility |
|---|---|
| `realworld_spiritai_eval_pi05_full_deploy` | overall → Flow, with hierarchy explicitly disabled |
| `realworld_spiritai_eval_pi05_subtask_deploy` | overall → generated Subtask → Flow; retain EOS checks and episode cache reset |
| `_spiritai_pi05_deploy.yaml` | Three images, BF16/FP32 islands, nominal 30 Hz, horizon 50, ten Euler steps, observer default, bounded RTC |
| `prepare_deployment.py` / `validate_deployment` | Record training provenance and weight SHA256; check mode, dimensions, state/text contract, and statistics |
| `check_spiritai.py` | Read-only preflight that releases the exclusive connection |
| `prepare_spiritai_server.py` | Verify exact source versions and create a new reviewed compatibility copy |
| Runner / env worker | Deliver driver stop requests over a native Channel; close the robot connection before waiting for rollout shutdown; discard final predictions |
| `SpiritaiCartesianEnv.close` | Attempt to release acquired following after execution errors; whole-body profiles reject endpoints that lose torso commands |
| `Pi0Eval.predict_action_batch` | Report actual sampler guidance in both Full and hierarchical paths |

Historical training settings and weights are unchanged. Full uses the established overall-only baseline. LoRA references the existing merged export, preserving historical metadata's `is_lora=true` while runtime `is_lora=false`. Sampling, training objectives, and model layers remain RLinf's implementations.

## Prepared Local Assets

Each bundle under ignored `model_checkpoints/` contains only `openpi_deployment.json` and `norm_stats.json`. Weights stay at their source paths. Startup verifies the complete weight SHA256, adding checkpoint I/O.

| Mode | Bundle | Weights |
|---|---|---|
| `flow_only` | `model_checkpoints/spiritai_deploy/flow_only` | `logs/20261028_095838-spiritai_foldbox_pi05_full_bs64/spiritai_foldbox_pi05_full_bs64/checkpoints/global_step_45000` |
| `explicit_subtask` | `model_checkpoints/spiritai_deploy/explicit_subtask` | `logs/20261102_233335-pi05_subtask_lora_bs64/offline_eval_20261110_163331/exports/step_45000` |

Full provenance explicitly records reconstruction from final W&B settings and saved loader signatures. LoRA checks W&B against embedded training metadata. Preserve both source weights and bundles when moving machines; the bundle alone does not contain a model. Recorded historical Git HEADs do not establish clean working trees.

The original `/home/deng/Documents/codebase/robot_server_code/` is unchanged. A compatibility copy at `/tmp/rlinf-spiritai-bridge-20261118/` was used for actual server/interpolation checks with a fake SDK. It is not a deployment dependency: regenerate a new copy from supported source with `prepare_spiritai_server`. The running robot container version still needs on-site comparison and manual synchronization.

## Commands for This Workstation

Run one of these commands from the repository root. The default reads robot observations and runs inference without sending actions or changing following. `cluster.component_placement.rollout=1` chooses RLinf hardware rank 1 for one model; use 0 to select the other rank.

Full Flow:

```bash
PATH="$PWD/.venv/bin:$PATH" bash evaluations/run_eval.sh realworld realworld_spiritai_eval_pi05_full_deploy \
  cluster.component_placement.rollout=1 \
  rollout.model.model_path="$PWD/logs/20261028_095838-spiritai_foldbox_pi05_full_bs64/spiritai_foldbox_pi05_full_bs64/checkpoints/global_step_45000" \
  rollout.model.openpi.deployment_metadata_path="$PWD/model_checkpoints/spiritai_deploy/flow_only/openpi_deployment.json" \
  rollout.model.openpi_data.norm_stats_path="$PWD/model_checkpoints/spiritai_deploy/flow_only/norm_stats.json"
```

Subtask LoRA:

```bash
PATH="$PWD/.venv/bin:$PATH" bash evaluations/run_eval.sh realworld realworld_spiritai_eval_pi05_subtask_deploy \
  cluster.component_placement.rollout=1 \
  rollout.model.model_path="$PWD/logs/20261102_233335-pi05_subtask_lora_bs64/offline_eval_20261110_163331/exports/step_45000" \
  rollout.model.openpi.deployment_metadata_path="$PWD/model_checkpoints/spiritai_deploy/explicit_subtask/openpi_deployment.json" \
  rollout.model.openpi_data.norm_stats_path="$PWD/model_checkpoints/spiritai_deploy/explicit_subtask/norm_stats.json"
```

Both inherit `ws://172.16.0.30:8766` and the FoldBox overall instruction. Append `env.eval.override_cfg.robot_url=ws://ROBOT_HOST:8766` to change the endpoint. Run this read-only check first and let it release its connection before starting the bridge:

```bash
.venv/bin/python -m toolkits.realworld_check.check_spiritai \
  --url ws://172.16.0.30:8766 --require-execution
```

The local old server does not advertise the required execution capabilities, so strict execution checks should reject it. The container must implement single-row targets, tracked busy/error state, and cancellation. After checking deployed code, camera orientation, SDK pose/gripper units, and frames, the operator can append `env.eval.override_cfg.execution_enabled=true` to the selected command. That profile starts after warmup and has no keyboard start gate.

W&B remains enabled. Warmup/bootstrap has a 30-second budget per request, while replanning has 1.2 seconds. Measure both policies on the actual GPU and network, especially LoRA HL cache misses. `runner.rtc.enabled` selects asynchronous scheduling; `rollout.model.openpi.rtc_enabled` independently selects sampler guidance.

## Verification and Limits

Automated checks use an RTX PRO 5000 Blackwell, the existing `.venv`, recorded observations, and actual server code bound only to loopback. The SDK and cameras are fake external boundaries; no robot is connected. See [implementation_validation.json](artifacts/implementation_validation.json) for results and final source fingerprints.

CPU checks cover training provenance/weight identity, configuration mode and statistics mismatches, read-only connection release, whole-body dimensions, following release after execution errors, and RTC clocks/stale reply rejection. GPU/native-worker checks run one model at a time on one GPU, covering both policies, observer/execution modes, disabled guidance, warmup/replan deadlines, driver SIGINT, and SDK errors after ACK.

The checks passed 26 relevant CPU unit cases and 12 protocol/native-runtime cases. Both policies completed separate 300-tick observation and simulated execution checks. Fault cases covered warmup/replan deadlines, SIGINT to the driver and its process group, and post-ACK SDK errors. Ruff, formatting, diff whitespace, and both Sphinx builds passed. The symbol checker retains 26 existing unresolved RST references, with no new findings.

| Soak | Ticks | First-to-last tick interval | Effective mean tick rate | Replan P99 | abort / close_error |
|---|---|---|---|---|---|
| Full Flow | 18050 | 710.10 seconds | 25.42 Hz | 0.272 seconds | 0 / 0 |
| Subtask LoRA | 18050 | 703.98 seconds | 25.64 Hz | 0.816 seconds | 0 / 0 |

LoRA P99 was 0.267 seconds for planner cache hits and 0.826 seconds when generating a Subtask. All 462 high-level generations ended with EOS. The fixed observation produced one Subtask, so this does not test stage transitions. GPU memory samples were 7458 MiB for Full and 7464 MiB for LoRA, with only one model loaded at a time; usage returned to baseline after exit.

`source_hz=30` defines the nominal action timeline and configuration. The complete loop measured about 25–26 Hz. Env-step pacing, RPC, worker processing, and logging lie on that path, so GPU latency is not the control period; individual overheads were not profiled. Remeasure the actual network and tick rate on site rather than claiming hard real-time 30 Hz.

Soak checks use a fixed recorded observation and fake SDK. They establish resource and communication stability, not physical folding success. Subtask text, EOS, and finite actions establish that the inference interface works; different box models still need real closed-loop evaluation.

## Diagnosis and Recovery

Read `logs/<timestamp>-<config>/eval_embodiment.log` and `<experiment_name>/rtc-env-0.jsonl` inside it. Locate `abort`, `hold`, and `closed` / `close_error`, then compare request/inference latency, Subtask/EOS, actual guidance, action index, and limits.

| Symptom | Check and response |
|---|---|
| Metadata, geometry, statistics, or SHA256 mismatch | Check the selected policy, 45000-step path, and bundle; retain validation |
| Missing capabilities or single-row interpolation failure | Inspect the running container; prepare supported source and rerun protocol checks |
| Replan timeout or exhausted chunk | Measure HL+LL, RPC, actual tick time, and remaining action horizon; do not repeat the last target indefinitely |
| Ctrl-C or SDK failure | Expect bounded hold attempts and connection cleanup; cooperative cancellation is not a motor emergency stop |
| Historical command lacks deployment manifest | Use the new deployment profiles with the correct bundle; archived commands retain their historical meaning |

Operators continue to launch formal training and real robot execution manually. This update has not been committed or pushed.
