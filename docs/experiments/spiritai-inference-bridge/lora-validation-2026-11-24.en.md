# Explicit Subtask LoRA Robot Inference Closeout

This experiment paused on 2026-11-24. The 45000-step LoRA checkpoint ran on one GPU through explicit subtask generation, flow matching, RTC, and physical action dispatch. Five-step sampling completed a 300-tick hardware check. The 1500-tick run ended after 1452 ticks. The operator reported excessive gripper contact with the table, joint overload, an emergency stop, and jerkier motion than with a previously used training-time RTC model. Closed-loop stage switching, complete folding, and motion smoothness remain unverified. [中文](lora-validation-2026-11-24.md)

## Recover the Model and Offline Evaluation

Use the `realworld_spiritai_eval_pi05_subtask_deploy` profile. The rollout worker loads merged LoRA weights. `is_lora=false` means inference loads the merged parameters directly. One workstation GPU runs both stages, while the robot uses the existing `robot-server/0.3.1-rlinf-compat` service. The model loader validates the weights, normalization, and deployment manifest.

| Asset | Local path |
|---|---|
| Merged 45000-step checkpoint | `logs/20261102_233335-pi05_subtask_lora_bs64/offline_eval_20261110_163331/exports/step_45000/` |
| Deployment manifest and training normalization | `model_checkpoints/spiritai_deploy/explicit_subtask/{openpi_deployment.json,norm_stats.json}` |
| LeRobot v2.1 test set | `/home/deng/Documents/dataset/20261109_FoldNJBox_Moz1_subtask_lora-test/` |
| Per-frame predictions and summary | `subtask_eval_20261123_095439_1daxGq/results/` under the training run |

Run `.venv/bin/python -m toolkits.standalone_eval_scripts.openpi.spiritai_subtask_eval` with `--dataset-root`, `--checkpoint`, `--deployment-metadata`, `--norm-stats`, and a new `--output`. Pass `--phase-map` to score coarse phases using an annotation map. The run saved its map as `phase_map.json`. The model receives three images, robot state, and the overall instruction. Reference subtasks determine sample selection and score the generated output.

All 1516 sampled frames across 80 episodes produced nonempty text and EOS. Phase agreement was 791/798 (99.1%) at segment midpoints and 523/720 (72.6%) in boundary windows. Agreement about 0.1 seconds before and after the annotation boundary was 68.8% and 55.0%, respectively. These results measure current-stage prediction from recorded observations. Closed-loop planning also requires completing the subtask and choosing the next one after the scene changes.

## Verified Hardware Settings

Keep the 50-action chunk, nominal 30 Hz, and `planning_interval=3`. Override both `rollout.model.num_steps=5` and `rollout.model.openpi.num_steps=5` for five Euler steps. Enable `runner.rtc.enabled=true` and `rollout.model.openpi.rtc_enabled=true`, use `approx` sampler guidance, and keep `runner.rtc.request_timeout_s=1.2`. For prediction-only checks, set `env.eval.override_cfg.execution_enabled=false`. See the [shared parameter reference](parameters.en.md) for each field. The hardware runs disabled the optional W&B logger.

| Run | Ticks completed / limit | Slowest planner refresh request | Result |
|---|---:|---:|---|
| Ten-step hardware smoke test | 50 / 50 | No accepted refresh replan | Dispatched physical actions |
| Ten-step hardware run | 253 / 300 | Accepted requests up to 1.154 s, request 17 timed out | Two holds, then closed |
| Five-step hardware run | 300 / 300 | 0.735 s | 11.64 s of control, zero aborts |
| Five-step previews and pose checks | 300 / 300 and three runs of 50 / 50 | 0.751 s | Repeated the alignment subtask |
| Five-step closed-loop long run | 1452 / 1500 | 0.756 s | 55.95 s of control, operator reported an emergency stop and software logged SIGINT |

Keep five steps as the tested timing configuration. High-level latency also differed across the runs, so the timing difference cannot be attributed entirely to fewer low-level steps. A paired check on 40 recorded samples held the observation, subtask, and noise fixed. The respective five-step and ten-step mean arm position errors over the first 10 rows were 17.0 and 18.1 mm. Mean 50-row rotation errors were 10.09° and 9.68°. This supported a short hardware check. Use physical observations and controller data to assess motion quality. The evidence archive contains paths, hashes, and statistics for all eight traces. [Hardware evidence](artifacts/realworld_explicit_subtask_20261124.json)

## Long-Run Incident and Stage Prediction

The long run accepted 40 newly generated subtasks including bootstrap. Every one requested carton alignment. The model also stayed on alignment in the read-only trial after the operator aligned the box and adjusted the arms. In an offline input swap, replacing recorded state with the hardware pose changed two boundary images back to alignment. All 24 images from the middle of the later stage still produced the correct stage. The small boundary sample and the arms visible in the images limit the result to pose sensitivity on some boundary inputs.

The operator reported excessive gripper contact with the table, joint overload, and a hardware emergency stop. The RLinf trace reports `InterruptedError: RTC driver requested stop: SIGINT`, followed by two holds and `closed`. The run has no synchronized joint torque, contact force, or protection telemetry. The software shutdown events do not verify mechanical stopping or recovery. The current per-row limits constrain target changes relative to measured state and clamp gripper commands. Their contract covers neither table collision nor contact forces. The experiment is paused. Check tabletop clearance, contact risk, and the protection state before resuming physical execution.

## Evidence About Jerky Motion

The operator reported that this run was jerkier than the previously used training-time RTC model. Median env step spacing was 37.88 ms, P99 was 49.28 ms, and the maximum was 53.77 ms. The slowest planner refresh request took 0.756 s while the control loop continued consuming the old chunk. **Request latency and control step spacing measure different things.** The trace shows no repeated 0.75 s control gap on refresh. The 25.93 Hz average alone cannot establish smoothness.

The time-based index skipped one model row on 230 consecutive-step transitions within a chunk. For the left wrist, the median translation change between selected targets across chunk boundaries was 536 mm before limiting and 1.32 mm after limiting. Within chunks, the corresponding values were 260 mm and 1.10 mm. These statistics describe selected destinations and limited commands. Assess controller velocity, acceleration, and joint jerk with separate telemetry. The larger target changes across chunks and their interaction with limiting are useful leads.

The RLinf profile uses inference-time `approx` guidance to steer samples toward remaining targets from the previous chunk. Each env tick limits the target relative to measured state, sends a single row, waits for SDK dispatch, and reads another observation. The local OpenPI reference at `origin/rct-rebuild@086bfa1` uses a fixed action prefix in its `training_time_v1` policy. Its `examples/spirit-ai/main.py` also sends one row at a time, with filters such as blending based on accepted command history, and waits for ACK at that dispatch point. The reference differs in both sampling and execution, so the field observation cannot isolate training-time RTC. Differences may come from predicted action continuity, chunk replacement and skipped rows, sequential RPCs, or controller tracking. The runs lack paired evidence from the same box, starting pose, and execution settings. Before comparing the two RTC policies, collect synchronized selected targets, dispatched commands, measured trajectories, torque, and protection events. Evaluate stage recognition and contact risk separately.
