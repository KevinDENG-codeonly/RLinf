# Full Flow 实现与验证记录

恢复 Full Flow 真机推理的组件职责、资产来源、部署变更和已验证边界。日常命令见[操作手册](README.md)，手动设置见[参数说明](parameters.md)，机器迁移与故障处理见[部署说明](deployment.md)。[English](implementation.en.md)

本轮软件基于 `main:f1d402ae`，工作分支为 `feature/spiritai-unified-inference-bridge`。2026-11-18 完成软件与模拟检查；2026-11-19 由用户手动完成机器人源码部署和 Full Flow 真机检查。训练、机器人动作和服务重启仍由操作人员手动启动。本文新增真机记录仅覆盖 Full Flow。

## 原生 RLinf 推理链路

工作站一次加载一个模型到一张 GPU。机器人 Docker 保持独立服务，不作为 Ray node；env worker 是唯一连接所有者。

| 代码或配置 | 实际职责 |
|---|---|
| `evaluations/run_eval.sh` / `eval_embodied_agent.py` | 时间戳日志、Hydra 配置、Cluster/placement 和 worker 启动 |
| `EmbodiedEvalRunner` | 初始化 worker、汇总指标、bounded RTC 停止通知与收尾等待 |
| `RTCEnvWorker` / `RTCSession` | 一次一行控制、单个在途请求、session/chunk 身份、monotonic action index、预算与 trace |
| `RTCMultiStepRolloutWorker` | 持有模型、串行预测、保留 model-space tail、返回实际 guidance 和时延 |
| `Pi0Eval` | 三路图像 + state + overall → 现有 Flow sampler → 动作解码 |
| `SpiritaiCartesianEnv` | 观测打包、基于测量状态限幅、following 生命周期、执行开关 |
| `SpiritaiClient` | NumPy/MsgPack、WebSocket、request ID、RPC 超时、状态/执行/取消协议 |
| `_spiritai_pi05_deploy.yaml` | 公共部署设置；Full primary 额外指定 `flow_only` 与 hierarchy=false |

50×32 的模型动作保留作 RTC overlap；解码到 50×25 后，bridge 按 tick 选择目标，删除两个 psi 后发送 23D。保留 torso，base 强制零。HL/LL 重建等既有功能不改变本次 Full Flow 的 overall-only 推理定义。

## 权重与训练来源

Full Flow 使用原生 consolidated checkpoint，模型 loader 直接解析目录，准备部署记录时不转换为其他 OpenPI 权重布局，也不重写训练权重。

| 资产 | 本机位置 |
|---|---|
| 45000-step 原始 checkpoint | `logs/20261028_095838-spiritai_foldbox_pi05_full_bs64/spiritai_foldbox_pi05_full_bs64/checkpoints/global_step_45000` |
| 权重 | 上述目录的 `actor/model_state_dict/full_weights.pt`，667 个 tensor entry |
| 最终 W&B config / launch metadata | 运行根目录下 `wandb/wandb/run-20261113_104028-8jehu52t/files/` |
| 部署 bundle | `model_checkpoints/spiritai_deploy/flow_only/openpi_deployment.json` 与 `norm_stats.json` |
| normalization 原始身份 | 与 `actor/data.pt` 中每个 rank 保存的 loader signature 一致 |

`prepare_deployment.py` 从最终 W&B config 与保存的 loader signature 重建 provenance，明确标记为重建记录，不伪造训练时缺失的 `openpi_training.json`。启动时核对模型 geometry、state/text 契约、normalization 内容和完整权重 SHA256。推理参数为 BF16，并保留 vision stem、norm 和 action projection 等既有 FP32 islands。

训练记录中的 `num_steps=5` 和当前推理采样 10 次分别保留；推理时不要求重新训练。历史训练 task 样例包含 overall + current step，但本实验既定 Full Flow 推理只使用 overall，不注入 GT Subtask。记录到的 Git HEAD 不能单独证明历史工作区干净。

## 软件阶段的历史证据

2026-11-18 的检查保持原始记录，不与后续真机结果混为一谈。[当时实施记录](implementation-2026-11-18.md)和[implementation_validation.json](artifacts/implementation_validation.json)覆盖 26 个相关 CPU 单测、12 项协议/原生链路检查，以及使用真实 server 代码、模拟 SDK 和固定录制观测的持续运行。该阶段没有连接真机。

Full Flow 的模拟持续检查为 18050 tick，710.10 秒，25.42 Hz，无 abort/close_error；重规划 P99 0.272 秒。相关故障检查覆盖 warmup/replan 超时、driver 和进程组 SIGINT、ACK 后 SDK 异常、单行动作、真实 Future busy、取消后停止后续提交。原 artifact 中其他实验的模拟记录按原时间与范围保留，此处不新增其真机结论。

Ruff 与双语 Sphinx 构建通过。全仓符号检查仍有 26 处既有 RST 引用问题，不在本次改动范围；检查结果不应描述成“所有仓库检查均通过”。

## 2026-11-19 的部署与真实运行

用户先用原协议读通机器人状态和图像。严格检查因缺少新字段失败后，审查 OpenPI 两个分支，确认新 `execution_capabilities` 属于 RLinf 适配约定。随后由用户提供实际 container 的 SHA256，与本地原始源码匹配；应用对应两文件补丁并通过 before/after 校验后，严格 preflight 返回 `robot-server/0.3.1-rlinf-compat`、rotvec、23D、三项能力 true、三路 240×320 RGB。

Full Flow 的三轮真实运行结果来自本机日志：

| 模式 | Tick | 首末 step 间隔 | 实测频率 | 重规划模型 / 请求 P99 | abort / close_error |
|---|---:|---:|---:|---|---|
| 只读预览 | 300 | 11.67 秒 | 25.62 Hz | 0.263 / 0.313 秒 | 0 / 0 |
| 真机短测 | 50 | 1.95 秒 | 25.19 Hz | 0.277 / 0.323 秒 | 0 / 0 |
| 约一分钟执行 | 1500 | 58.97 秒 | 25.42 Hz | 0.269 / 0.321 秒 | 0 / 0 |

三轮 trace 均以 `closed` 结束，记录到的预测和限幅动作均为有限值。用户确认 50 tick 时机器人运动，并在 1500 tick 后确认真机推理链路正常。35、7、168 个 response 分别包含 warmup/bootstrap 与 replan；不能把请求计数当作实际动作行数。详细统计、trace 路径与 hash 见[realworld_full_flow_20261119.json](artifacts/realworld_full_flow_20261119.json)。

本次 W&B 报错来自遗漏训练时的自定义 endpoint，发生于权重加载前。操作命令现显式关闭可选云端后端；需要 W&B 时使用 `WANDB_BASE_URL=https://api.forge.coreweave.com`。网络、Docker 导入、源码包和协议纠正过程已整理到[部署说明](deployment.md)。

## 结论边界与后续对照

这些结果证明 Full Flow 从训练权重、真实观测、RTC 重规划到动作下发与软件关闭的链路已跑通。它们不构成量化折盒成功率、不同盒型泛化率或与其他实验的性能优劣结论。没有新增训练、修改数据集、引入 KI/FAST 或改变模型层结构。

标称 `source_hz=30` 保持数据时间解释，实测闭环约 25–26 Hz；首末 tick 时长不含加载和预热。runtime 仍按步数结束。改变采样步数、RTC、速度限幅、overall、相机条件或初始状态时，记录实际 resolved config 和观察结果；重新分离任务表现与工程吞吐。

following 开启不等于已发送动作；Future 结束不等于机械到位；软件取消不清除已接受的底层目标。真实运动方向、夹爪指令尺度和现场任务结果仍由操作人员观察。当前环境没有自动任务成功判定，因此不能把零 reward/success 当作任务失败。
