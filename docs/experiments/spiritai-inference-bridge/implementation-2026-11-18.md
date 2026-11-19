# SpiritAI 统一推理 Bridge：实施记录

这是 2026-11-18 软件阶段的历史快照，保留当时的部署状态和命令。当前 Full Flow 操作入口见 [README](README.md)。

使用现有 RLinf evaluation 链路，在 Full Flow 与 Explicit Subtask LoRA 之间选择一种 policy，每次加载一个模型、使用一张 GPU。本文记录实现、部署资产、验证范围和本机启动命令；设计依据见[获批方案](research-proposal.md)。[English](implementation.en.md)

实施与验证日期：2026-11-18。

## 实现与职责

工作分支为 `feature/spiritai-unified-inference-bridge`，基于 `f1d402ae`。启动入口仍为 `evaluations/run_eval.sh`，依次使用 `EmbodiedEvalRunner`、`RTCEnvWorker`、`RTCMultiStepRolloutWorker`、`Pi0Eval`、`SpiritaiCartesianEnv` 与 `SpiritaiClient`。只有 env 侧连接机器人 container。两种 policy 分开启动，共用相机、动作解码、限幅、RTC 调度与通信。

| 入口或文件 | 作用 |
|---|---|
| `realworld_spiritai_eval_pi05_full_deploy` | overall → Flow；显式关闭 hierarchy |
| `realworld_spiritai_eval_pi05_subtask_deploy` | overall → 生成 Subtask → Flow；保留 EOS 检查与 episode cache 重置 |
| `_spiritai_pi05_deploy.yaml` | 共用三图、BF16/FP32 islands、标称 30 Hz、50 行 chunk、10 次 Euler、只读默认与 bounded RTC |
| `prepare_deployment.py` / `validate_deployment` | 保存训练来源与权重 SHA256；核对模式、维度、state/text 设置和 stats |
| `check_spiritai.py` | 只读 preflight，检查完成后释放独占连接 |
| `prepare_spiritai_server.py` | 检查精确源码版本，在新目录生成已审阅的兼容副本 |
| runner / env worker | driver 停止通知通过原生 Channel 传递；关闭连接后等待 rollout 结束，最后预测不阻挡关闭 |
| `SpiritaiCartesianEnv.close` | 执行错误后仍尝试关闭本次 env 开启的 following；whole-body 配置拒绝丢弃 torso |
| `Pi0Eval.predict_action_batch` | Full 和 hierarchical 路径都报告实际 sampler guidance，而非请求层的推测 |

历史训练配置与权重未修改。Full 仍采用既定 overall-only 推理条件；LoRA 使用已有合并导出，原始 metadata 中的 `is_lora=true` 保留，运行时为 false。模型的 sampler、训练目标和层结构沿用 RLinf。

## 已准备的本地资产

两个 bundle 已生成在忽略的 `model_checkpoints/` 中，每个目录只包含 `openpi_deployment.json` 与 `norm_stats.json`，权重仍从原路径加载。manifest 记录完整权重文件 SHA256，首次加载会产生相应 I/O。

| 模式 | 部署 bundle | 权重 |
|---|---|---|
| `flow_only` | `model_checkpoints/spiritai_deploy/flow_only` | `logs/20261028_095838-spiritai_foldbox_pi05_full_bs64/spiritai_foldbox_pi05_full_bs64/checkpoints/global_step_45000` |
| `explicit_subtask` | `model_checkpoints/spiritai_deploy/explicit_subtask` | `logs/20261102_233335-pi05_subtask_lora_bs64/offline_eval_20261110_163331/exports/step_45000` |

Full 的 provenance 明确标记为从最终 W&B 配置与保存的 loader signature 重建；LoRA 同时核对 W&B 和 checkpoint metadata。移动机器时保留 source weights 与两个 bundle；只移动 bundle 不会复制模型。历史 Git HEAD 不能证明当时工作区干净。

本地 `/home/deng/Documents/codebase/robot_server_code/` 未改动。兼容副本生成于 `/tmp/rlinf-spiritai-bridge-20261118/`，用于实际 server/插值代码与模拟 SDK 的测试；它不是正式部署依赖。可用 `prepare_spiritai_server` 命令从已支持源码重新生成新副本。机器人上实际运行版本仍待现场核对和人工同步。

## 本机启动命令

在仓库根目录执行下列两个命令中的一个。默认只读取机器人观测并运行 policy，不发送动作，也不切换 following。`cluster.component_placement.rollout=1` 选择 RLinf hardware rank 1，只加载一个模型；可改为 0。

Full Flow：

```bash
PATH="$PWD/.venv/bin:$PATH" bash evaluations/run_eval.sh realworld realworld_spiritai_eval_pi05_full_deploy \
  cluster.component_placement.rollout=1 \
  rollout.model.model_path="$PWD/logs/20261028_095838-spiritai_foldbox_pi05_full_bs64/spiritai_foldbox_pi05_full_bs64/checkpoints/global_step_45000" \
  rollout.model.openpi.deployment_metadata_path="$PWD/model_checkpoints/spiritai_deploy/flow_only/openpi_deployment.json" \
  rollout.model.openpi_data.norm_stats_path="$PWD/model_checkpoints/spiritai_deploy/flow_only/norm_stats.json"
```

Subtask LoRA：

```bash
PATH="$PWD/.venv/bin:$PATH" bash evaluations/run_eval.sh realworld realworld_spiritai_eval_pi05_subtask_deploy \
  cluster.component_placement.rollout=1 \
  rollout.model.model_path="$PWD/logs/20261102_233335-pi05_subtask_lora_bs64/offline_eval_20261110_163331/exports/step_45000" \
  rollout.model.openpi.deployment_metadata_path="$PWD/model_checkpoints/spiritai_deploy/explicit_subtask/openpi_deployment.json" \
  rollout.model.openpi_data.norm_stats_path="$PWD/model_checkpoints/spiritai_deploy/explicit_subtask/norm_stats.json"
```

两者沿用本机配置的 `ws://172.16.0.30:8766` 与 FoldBox overall。可追加 `env.eval.override_cfg.robot_url=ws://ROBOT_HOST:8766` 替换地址。先运行下面的只读检查，检查结束后再启动 bridge，以免争用唯一连接：

```bash
.venv/bin/python -m toolkits.realworld_check.check_spiritai \
  --url ws://172.16.0.30:8766 --require-execution
```

当前本地旧版 server 不声明所需执行能力，严格检查应拒绝该版本；需要匹配实现单行目标、真实 busy/错误、取消协议的 container。完成现场版本、相机方向、SDK 位姿/夹爪单位与参考系检查后，操作人员才在所选命令末尾追加 `env.eval.override_cfg.execution_enabled=true`。该配置预热后直接执行，不含键盘启动 gate。

W&B 保留。bootstrap/warmup 为每请求 30 秒，replan 为 1.2 秒；分别校验两类模型在实际 GPU 与网络上的时延，尤其是 LoRA 的 HL cache miss。`runner.rtc.enabled` 控制异步调度，`rollout.model.openpi.rtc_enabled` 单独控制 sampler guidance。

## 验证与限制

自动化验证使用 RTX PRO 5000 Blackwell、当前 `.venv`、录制观测和仅绑定 loopback 的真实 server 代码。`pymozrobot` 与相机位于模拟边界，未连接机器人。结果与最终源码指纹见 [implementation_validation.json](artifacts/implementation_validation.json)。

CPU 检查覆盖训练 provenance/权重身份、配置模式与 stats 不匹配、只读连接释放、whole-body 维度、执行错误后的 following 释放、RTC session 时钟和旧回复拒绝。GPU/原生 worker 检查按一张卡、一个模型依次运行，验证两种 policy、预览/执行、guidance 关闭、warmup/replan 超时、driver SIGINT 和 ACK 后 SDK 错误。

已完成 26 个相关 CPU 单测和 12 项协议/原生链路检查。两种 policy 均单独完成 300 tick 的预览与模拟执行；故障检查覆盖 warmup 超时、replan 超时、driver/进程组 SIGINT 和 ACK 后 SDK 错误。Ruff、格式检查、diff whitespace 与 EN/ZH Sphinx 构建通过；全仓符号检查保留 26 处既有 RST 引用问题，未新增。

| 持续检查 | Tick 数 | 首末 tick 间隔 | 实际平均 tick 率 | 重规划 P99 | abort / close_error |
|---|---|---|---|---|---|
| Full Flow | 18050 | 710.10 秒 | 25.42 Hz | 0.272 秒 | 0 / 0 |
| Subtask LoRA | 18050 | 703.98 秒 | 25.64 Hz | 0.816 秒 | 0 / 0 |

LoRA 缓存命中的 P99 为 0.267 秒，重新生成 Subtask 时为 0.826 秒；462 次高层生成均以 EOS 结束。固定观测下只生成一种 Subtask，不用于判断阶段迁移能力。GPU 显存抽样为 Full 7458 MiB、LoRA 7464 MiB；每次只有一份模型，退出后 GPU 占用恢复基线。

`source_hz=30` 是动作时间轴与配置的标称频率，本次完整闭环实测约 25–26 Hz。env step 的等待、RPC、worker 处理与日志均处于控制路径，不能把 GPU 推理时延当成实际 tick 间隔；未对各项开销分别归因。现场需重新测量网络和实际 tick，不能据此宣称达到硬实时 30 Hz。

持续运行使用固定录制观测和模拟 SDK，验证资源与通信稳定性，不表示完成真实折盒。Subtask 文本、EOS 和有限动作证明推理接口可工作，不能替代不同型号盒子的真机闭环成功率。

## 故障定位与恢复

查看 `logs/<启动时间>-<config>/eval_embodiment.log` 和其中 `<experiment_name>/rtc-env-0.jsonl`。先定位 `abort`、`hold`、`closed` / `close_error`，再检查请求/实际推理时延、Subtask/EOS、guidance、action index 与限幅。

| 现象 | 检查与处理 |
|---|---|
| metadata、维度、stats 或 SHA256 不匹配 | 核实所选 policy、45000-step 路径与对应 bundle；不要关闭校验绕过错误 |
| 执行能力不足或单行插值报错 | 核对运行中的 container；用受支持源码生成兼容副本并重新做协议检查 |
| replan 超时、chunk 耗尽 | 检查 HL+LL、RPC 与 tick 的真实耗时及剩余动作时间；不能无限复用最后目标 |
| Ctrl-C / SDK 报错 | 应先出现有限 hold 尝试和关闭记录；软件取消只停止后续下发，不等价电机急停 |
| 历史命令缺 deployment manifest | 使用新的两套 deployment 配置，并传入所选 bundle；旧实验文档中的命令保留历史语义 |

正式训练与真机执行仍由操作人员手动启动。本次更新没有提交或推送远端。
