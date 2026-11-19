# SpiritAI Full Flow 真机推理操作手册

从这里启动已经验证的 Full Flow + RTC 真机链路，选择只读预览或动作执行，并查明每项手动参数的作用、默认值和修改位置。每次运行一个模型、使用一张 GPU；正式训练与真机动作命令由操作人员手动启动。[English](README.en.md)

截至 2026-11-19，45000-step Full Flow 已完成真实观测预览、50 tick 真机短测和 1500 tick 真机执行；用户确认真机推理链路正常。下表是本机日志验证结果，不表示量化折盒成功率。

| 运行 | Tick | 实测平均频率 | abort / close_error | 结果 |
|---|---:|---:|---|---|
| 只读预览 | 300 | 25.62 Hz | 0 / 0 | 预测动作，未下发 |
| 真机短测 | 50 | 25.19 Hz | 0 / 0 | 用户确认机器人运动，正常关闭 |
| 约一分钟真机执行 | 1500 | 25.42 Hz | 0 / 0 | 首末 tick 间隔 58.97 秒，正常关闭 |

## 文档入口

按操作目的查阅下列页面。原调研方案已归档，日常操作以本页和参数说明为准。

| 文档 | 内容 |
|---|---|
| [推理参数说明](parameters.md) | 三个常用覆盖参数、完整参数表、YAML 修改位置、时长计算和参数优先级 |
| [机器人部署与故障处理](deployment.md) | 有线网络、Docker 镜像、源码挂载、兼容补丁、只读检查、W&B 和常见错误 |
| [实现与验证记录](implementation.md) | RLinf 组件职责、训练资产、历史模拟验证及 2026-11-19 真机验证边界 |
| [真机验证证据](artifacts/realworld_full_flow_20261119.json) | 三轮运行的日志位置、trace hash、RTC 延迟、动作与关闭检查 |
| [原研究方案](research-proposal.md) | 开发前的调研、差距和设计依据；保留历史状态 |

## 两台机器各运行什么

先在机器人启动 SDK container，再在工作站启动 RLinf。机器人连接由 env worker 独占；读完 preflight 后释放连接，再运行 policy。

```mermaid
flowchart LR
    P[工作站 / GPU 1 / Full Flow] <-->|RLinf Channel| E[工作站 / CPU / RTCEnvWorker]
    E <-->|WebSocket / 172.16.0.30:8766| S[机器人 / Python robot_server]
    S --> K[现有 pymozrobot / ROS2 / 控制器]
```

工作站使用 `evaluations/run_eval.sh → EmbodiedEvalRunner → RTC workers → Pi0Eval`。模型根据三路图像、25D state 和 overall 指令预测动作；Full Flow 关闭高层 Subtask Generation。50×32 的 model-space chunk 解码为 50×25，去掉两个 psi 后按 tick 发送一行 23D 指令。container 的 120 Hz 下发层与模型查询、标称 30 Hz 动作时间轴分别计时。

## 先确认机器人服务

机器人当前已部署 `robot-server/0.3.1-rlinf-compat`，image 仍为 `harbor.i.spirit-ai.com:443/shenming/thor-robot:0526`。源码校验与更新记录见[部署说明](deployment.md)。需要重新启动服务时，在机器人终端执行：

```bash
cd ~/robot_server_code
bash run_robot.sh
docker logs --tail 100 -f robot_server
```

等待 `listening on ws://0.0.0.0:8766`。查看日志时按 `Ctrl+C` 只退出日志跟随，后台容器继续运行。脚本重新运行会删除并重建同名 container，因此仅在准备启动或重启服务时使用。

在工作站检查实际 metadata、状态和三路图像：

```bash
cd /home/deng/Documents/codebase/RLinf
.venv/bin/python -m toolkits.realworld_check.check_spiritai \
  --url ws://172.16.0.30:8766 --require-execution
```

预期为 `cart_dim=23`、三路 `[240,320,3]` RGB 图像、`busy=false`、`execution_error=null`，以及三项执行能力均为 true。preflight 不发动作，也不切换 following；它报告的 `external_following_enabled` 可以已经为 true。这组能力字段是当前 RLinf 部署适配新增的约定，原 OpenPI bridge 没有要求它们；不能只依据字段缺失推断机器人底层不能工作，版本判断过程见[故障记录](deployment.md)。

## 运行真实观测的只读预览

在工作站仓库根目录使用下列完整命令，读取真实观测并预测 300 tick。权重和部署 bundle 已在本机准备完成；迁移机器时按[资产说明](implementation.md)保留对应文件。

```bash
PATH="$PWD/.venv/bin:$PATH" \
bash evaluations/run_eval.sh realworld realworld_spiritai_eval_pi05_full_deploy \
  cluster.component_placement.rollout=1 \
  rollout.model.model_path="$PWD/logs/20261028_095838-spiritai_foldbox_pi05_full_bs64/spiritai_foldbox_pi05_full_bs64/checkpoints/global_step_45000" \
  rollout.model.openpi.deployment_metadata_path="$PWD/model_checkpoints/spiritai_deploy/flow_only/openpi_deployment.json" \
  rollout.model.openpi_data.norm_stats_path="$PWD/model_checkpoints/spiritai_deploy/flow_only/norm_stats.json" \
  runner.rtc.enabled=true \
  rollout.model.openpi.rtc_enabled=true \
  rollout.model.openpi.rtc_guidance_mode=approx \
  env.eval.override_cfg.robot_url=ws://172.16.0.30:8766 \
  env.eval.max_steps_per_rollout_epoch=300 \
  env.eval.max_episode_steps=300 \
  env.eval.override_cfg.execution_enabled=false \
  'runner.logger.logger_backends=[]'
```

命令使用 GPU 1，开启 RTC 调度与 `approx` sampler guidance，保持 `execution_enabled=false`。即使 trace 出现 `step`，也只是预览控制步，不发送动作或切换 following。权重 SHA256 校验和加载可能花费数十秒；随后两轮 warmup 与 bootstrap 完成，才进入控制循环。首个 bootstrap 没有旧 chunk，guidance=false 正常；后续有效 replan 应为 true。

## 运行约一分钟的动作

在任务起始构型和盒子位置准备好后，手动执行下面的命令。`execution_enabled=true` 表示预热及 bootstrap 后直接开始动作，没有按键启动 gate；reset 读取当前位置，不自动回示范起始姿态。

```bash
PATH="$PWD/.venv/bin:$PATH" \
bash evaluations/run_eval.sh realworld realworld_spiritai_eval_pi05_full_deploy \
  cluster.component_placement.rollout=1 \
  rollout.model.model_path="$PWD/logs/20261028_095838-spiritai_foldbox_pi05_full_bs64/spiritai_foldbox_pi05_full_bs64/checkpoints/global_step_45000" \
  rollout.model.openpi.deployment_metadata_path="$PWD/model_checkpoints/spiritai_deploy/flow_only/openpi_deployment.json" \
  rollout.model.openpi_data.norm_stats_path="$PWD/model_checkpoints/spiritai_deploy/flow_only/norm_stats.json" \
  runner.rtc.enabled=true \
  rollout.model.openpi.rtc_enabled=true \
  rollout.model.openpi.rtc_guidance_mode=approx \
  env.eval.override_cfg.robot_url=ws://172.16.0.30:8766 \
  env.eval.max_steps_per_rollout_epoch=1500 \
  env.eval.max_episode_steps=1500 \
  env.eval.override_cfg.execution_enabled=true \
  'runner.logger.logger_backends=[]'
```

1500 是控制 tick 数，不是秒数、模型查询次数或训练更新次数。当前真机约 25.4 Hz，1500 tick 的动作阶段约 59 秒；模型加载、校验和预热另计。此配置按步数结束，不是严格的 60 秒墙钟计时器。正常结束会取消后续下发、尝试释放 following 并关闭连接；`Ctrl+C` 请求同一软件收尾流程，不能代替硬件急停。

## 为什么三个参数要写在命令里

它们是 Hydra 的运行配置覆盖项，不是送入模型的 observation。最终配置由 YAML defaults 逐层合并，最后应用命令行覆盖；若接受 YAML 默认值，可以省略这三项。

| 参数 | 当前部署默认值 | 示例为什么覆盖 |
|---|---|---|
| `env.eval.max_steps_per_rollout_epoch` | `3000`，来自环境模板 | 限制每轮评估的 worker 控制循环预算，改为 1500 以缩短运行 |
| `env.eval.max_episode_steps` | `3000`，来自环境模板 | 限制环境单 episode 长度，改为 1500 后在同一处产生截断 |
| `env.eval.override_cfg.execution_enabled` | `false`，来自公共部署片段 | 显式选择向机器人下发动作；保留 false 则是只读预览 |

两个长度上限不要求相等。当前 `rollout_epoch=1`、`auto_reset=false`，通常到较小上限先结束：前者控制外层循环，后者通过环境 truncation 产生 done。一起设置只是让本轮预算和 episode 边界一致。完整例子、整除规则和永久写入 YAML 的方法见[参数说明](parameters.md)。

本次整理只更新文档，保留现有 YAML 默认 `3000 / 3000 / false`。把运行长度写进 `realworld_spiritai_eval_pi05_full_deploy.yaml` 后，可省略对应命令行项；日常建议继续显式选择执行开关，使每条动作命令容易审阅。

## 读取结果与排查问题

`run_eval.sh` 为每次启动创建新目录，详细 trace 与终端日志分别位于：

```text
logs/<时间>-realworld_spiritai_eval_pi05_full_deploy/eval_embodiment.log
logs/<时间>-realworld_spiritai_eval_pi05_full_deploy/spiritai_pi05_full_deploy/rtc-env-0.jsonl
```

正常结果为 `rtc_aborted=0`、预期 tick 数、最后的 `closed`，没有 `close_error`。最后一条未完成预测可以记录 `response_discarded: episode_limit`，表示运行结束时丢弃预测，不是故障。`Evaluating RTC Rollout: 8it` 是请求处理计数，不是只发送了 8 行动作。当前环境没有折盒成功自动判定，`reward=0 / success=0` 不等于动作推理失败。

本页命令用 `runner.logger.logger_backends=[]` 关闭可选云端日志，终端与 RTC JSONL 仍保留。推理不依赖 W&B；需要 W&B 时，删除该覆盖并在启动前设置本机已有凭据对应的 endpoint：

```bash
export WANDB_BASE_URL=https://api.forge.coreweave.com
```

`No API key configured` 若发生在 `MetricLogger → wandb.init`，模型权重尚未加载；先核对 endpoint，不要因此重导出权重或在命令中写入密钥。网络、容器版本、执行能力、超时及恢复步骤见[部署与故障处理](deployment.md)。
