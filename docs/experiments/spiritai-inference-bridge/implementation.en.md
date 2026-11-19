# Full Flow Implementation and Verification

Recover component ownership, asset provenance, deployment changes, and verified limits of Full Flow robot inference. Use the [runbook](README.en.md) for commands, [parameters](parameters.en.md) for manual settings, and [deployment](deployment.en.md) for machine setup and failures. [中文](implementation.md)

Software is based on `main:f1d402ae`, on branch `feature/spiritai-unified-inference-bridge`. Software/simulation checks completed on 2026-11-18; the user manually deployed robot source and checked Full Flow on hardware on 2026-11-19. Operators launch training, robot actions and service restarts manually. New hardware results here cover Full Flow only.

## Native RLinf Runtime

One workstation loads one model on one GPU per run. The robot Docker service remains outside Ray, and the env worker owns the sole robot connection.

| Code or configuration | Responsibility |
|---|---|
| `evaluations/run_eval.sh` / `eval_embodied_agent.py` | Timestamp logs, Hydra, Cluster/placement and worker launch |
| `EmbodiedEvalRunner` | Initialization, metrics, bounded RTC stop notifications and cleanup waits |
| `RTCEnvWorker` / `RTCSession` | One-row control, one pending request, session/chunk identity, monotonic action index, budgets and trace |
| `RTCMultiStepRolloutWorker` | One model, serialized prediction, model-space tail, actual guidance and timing |
| `Pi0Eval` | Three images + state + overall → existing Flow sampler → decoded actions |
| `SpiritaiCartesianEnv` | Observations, measured-state limits, following lifecycle, execution switch |
| `SpiritaiClient` | NumPy/MsgPack, WebSocket, request IDs, RPC deadlines, status/dispatch/cancellation |
| `_spiritai_pi05_deploy.yaml` | Shared deployment settings; Full primary adds `flow_only` and hierarchy=false |

Retain 50×32 model actions for RTC overlap. Decode to 50×25, select one target per tick, remove two psi values, and send 23D. Torso stays present and base is zero. Existing HL/LL functionality does not change this Full Flow experiment's overall-only definition.

## Weights and Training Provenance

Full Flow loads a native consolidated checkpoint directly. Deployment preparation neither converts it to another OpenPI layout nor rewrites training weights.

| Asset | Local location |
|---|---|
| Original 45000-step checkpoint | `logs/20261028_095838-spiritai_foldbox_pi05_full_bs64/spiritai_foldbox_pi05_full_bs64/checkpoints/global_step_45000` |
| Weights | `actor/model_state_dict/full_weights.pt` under that directory, 667 tensor entries |
| Final W&B configuration/launch metadata | `wandb/wandb/run-20261113_104028-8jehu52t/files/` under the run root |
| Deployment bundle | `model_checkpoints/spiritai_deploy/flow_only/openpi_deployment.json` and `norm_stats.json` |
| Original normalization identity | Matches every saved rank's loader signature in `actor/data.pt` |

`prepare_deployment.py` reconstructs provenance from final W&B settings and saved loader signatures, labeling it as reconstructed instead of inventing a historical `openpi_training.json`. Startup checks model geometry, state/text contract, normalization content and complete weight SHA256. Inference uses BF16 with existing FP32 islands for vision stem, norms and action projections.

Keep recorded training `num_steps=5` separate from ten inference steps; inference does not require retraining. Historical task samples include overall/current step, while this experiment's established Full Flow inference uses overall only without GT Subtasks. A recorded Git HEAD alone does not establish a clean historical working tree.

## Historical Software Evidence

Preserve 2026-11-18 evidence separately from later hardware results. The [implementation snapshot](implementation-2026-11-18.en.md) and [implementation_validation.json](artifacts/implementation_validation.json) cover 26 relevant CPU cases, 12 protocol/native-runtime checks, and soak with actual server code, fake SDK and fixed recorded observations. No physical robot was connected in that stage.

Full Flow simulation completed 18050 ticks over 710.10 seconds at 25.42 Hz without abort/close_error; replan P99 was 0.272 seconds. Fault checks cover warmup/replan deadlines, SIGINT to the driver/process group, post-ACK SDK errors, one-row commands, Future-based busy, and cancellation of further submissions. Other experiments' historical simulated records remain unchanged, without adding hardware conclusions here.

Ruff and both Sphinx builds passed. The repository symbol checker retains 26 existing unresolved RST references outside this change; do not describe that as every repository check passing.

## Deployment and Hardware Runs on 2026-11-19

The user first read robot state/images through the original protocol. After strict checks failed on absent new fields, both OpenPI branches were reviewed to establish that `execution_capabilities` are RLinf adapter additions. User-provided hashes from the running container matched reviewed originals. The two-file patch passed before/after verification; strict preflight then reported `robot-server/0.3.1-rlinf-compat`, rotvec, 23D, three true capabilities and three 240×320 RGB views.

Three Full Flow runs are recorded locally:

| Mode | Ticks | First-to-last step span | Effective rate | Replan model / request P99 | abort / close_error |
|---|---:|---:|---:|---|---|
| Observation preview | 300 | 11.67 seconds | 25.62 Hz | 0.263 / 0.313 seconds | 0 / 0 |
| Short robot check | 50 | 1.95 seconds | 25.19 Hz | 0.277 / 0.323 seconds | 0 / 0 |
| Approximately one-minute run | 1500 | 58.97 seconds | 25.42 Hz | 0.269 / 0.321 seconds | 0 / 0 |

All traces end with `closed`, and recorded predicted/limited actions are finite. The user confirmed motion in the 50-tick run and a working hardware inference chain after 1500 ticks. The respective 35, 7 and 168 responses include warmup/bootstrap/replan; request counts are not action-row counts. See [realworld_full_flow_20261119.json](artifacts/realworld_full_flow_20261119.json) for statistics, paths and hashes.

W&B failed before weight loading because the command omitted the custom training endpoint. Current operating commands disable optional cloud metrics; retaining W&B requires `WANDB_BASE_URL=https://api.forge.coreweave.com`. See [deployment](deployment.en.md) for network, image import, source packaging and protocol corrections.

## Interpretation and Future Comparisons

These runs establish Full Flow's path from training checkpoint through real observations, RTC replanning, action dispatch and software shutdown. They do not quantify folding success, box-type generalization, or superiority over other experiments. No training, dataset changes, KI/FAST objectives or model layers were added.

Nominal `source_hz=30` retains the dataset timeline while measured control runs around 25–26 Hz. First-to-last tick spans exclude loading and warmup, and runtime remains step-limited. Record resolved settings and observations when changing sampling, RTC, limits, overall instruction, cameras or initial state; distinguish task performance from engineering throughput.

Following enabled does not imply dispatch. Future completion does not establish mechanical arrival, and cancellation cannot erase targets already accepted below the Python layer. Operators observe motion direction, gripper command scale and task outcomes. This env has no automatic folding judge, so zero reward/success is not evidence of task failure.
