# 实验 C：单 GPU 真机推理工程记录

使用本记录恢复实验 C 的部署配置，先检查预测与机器人协议，再由操作人员启动实际动作。当前部署沿用 RLinf PyTorch 模型和 Ray worker；机器人已有的 Docker container 负责图像、状态与底层执行。

## 部署结构与权重

GPU 工作站运行一个 `RTCMultiStepRolloutWorker` 和一个 `RTCEnvWorker`。前者加载完整 policy，在同一 GPU 上依次完成 HL Subtask Generation 和 LL Flow Sampling；后者承担 bridge 职责，通过 `SpiritaiCartesianEnv → SpiritaiClient` 访问机器人 WebSocket。机器人 container 本身不加入 Ray cluster。SpiritAI env 使用 `runtime_setup=false`，注册任务时跳过工作站上的 ROS 清理；本机直接驱动机器人的配置保留默认 setup。

入口配置为 `evaluations/realworld/realworld_spiritai_eval_pi05_subtask_deploy.yaml`。它与旧的普通 RTC、Subtask RTC 配置共享 `_spiritai_pi05_rtc.yaml` 和 `_spiritai_pi05_subtask.yaml`。Hydra 的 `searchpath` 只能声明在 primary config，因此共享配置不声明它，各入口分别声明。

推理使用合并后的完整权重，设置 `is_lora=false`。这表示 adapter 已合并进基础权重，训练仍然是 LoRA。部署目录保留 `full_weights.pt`、`openpi_training.json` 和 `norm_stats.json`。模型加载前检查训练 metadata 中的维度、state token、文本预算、相机数量、文本模板及 normalization；允许推理采样步数和规划频率不同于训练。

本机已完成的 45000-step 导出位于：

```text
logs/20261102_233335-pi05_subtask_lora_bs64/offline_eval_20261110_163331/exports/step_45000/
```

部署使用 BF16 backbone 与现有 FP32 islands，action horizon 50、模型维度 32、机器人维度 25、10 次 Euler integration。默认 `planning_interval=3` 指三个 action request；每个 request 可能执行不同数量的控制 tick，不能把它解释为三秒或三个 tick。

## 先运行观测模式

新配置默认 `execution_enabled=false`。它连接 container、读取相机和状态、运行 policy 并记录经过限幅的动作，但 client 只发送 `get_obs` 和 `get_status`，不切换 external following，也不发送动作。连接具有 container 的单客户端排他语义，请先退出其他 bridge。

在现有 venv 中运行：

```bash
cd /home/deng/Documents/codebase/RLinf
source .venv/bin/activate
PI05_DEPLOY_CKPT="$PWD/logs/20261102_233335-pi05_subtask_lora_bs64/offline_eval_20261110_163331/exports/step_45000"

CUDA_VISIBLE_DEVICES=0 bash evaluations/run_eval.sh realworld realworld_spiritai_eval_pi05_subtask_deploy \
  rollout.model.model_path="$PI05_DEPLOY_CKPT" \
  rollout.model.openpi_data.norm_stats_path="$PI05_DEPLOY_CKPT/norm_stats.json" \
  env.eval.override_cfg.robot_url=ws://172.16.0.30:8766
```

该命令选择一张 GPU，保留原有 W&B logger。启动后先完成两轮 warmup，涵盖普通 Flow 与 `approx` RTC 路径；随后重新读取观测并 bootstrap。每个 episode 都清空高层文本和动作缓存，即使 overall 没有变化。

日志位于 `logs/<时间>-realworld_spiritai_eval_pi05_subtask_deploy/`，其中 `spiritai_pi05_subtask_deploy/rtc-env-0.jsonl` 按 session、request 和 chunk 记录请求、生成文本、EOS、延迟、动作下发前后的值、测量状态及退出原因。缓存命中时不会重新生成 EOS，结合 `subtask_regenerated` 判断该条记录。

## 确认动作契约后开启执行

执行模式要求 container 的 hello metadata 声明以下能力：

```json
{
  "execution_capabilities": {
    "single_step_cartesian": true,
    "tracked_completion": true,
    "cancel_actions": true
  }
}
```

这些声明必须对应实际实现。`single_step_cartesian` 支持单行绝对目标；`tracked_completion` 根据后台任务完成状态报告 busy 并暴露执行错误；`cancel_actions` 取消后续 SDK 下发，并等待后台任务结束。

审查时本地 `robot_server_code` 存在三个问题：`T=1` 进入 PCHIP/SLERP 后报错；ACK 先于后台执行，busy 只按预计时间清除；`cancel_actions()` 是空实现。兼容补丁在 [robot_server_compat.patch](artifacts/robot_server_compat.patch)，原始和修改后文件 SHA256 见 [manifest](artifacts/robot_server_compat_manifest.json)。补丁仅在临时副本验证，尚未应用于机器人或原始源码。

先在实际部署源码上核对版本及补丁上下文。以下命令只检查补丁能否应用：

```bash
cd /home/deng/Documents/codebase/robot_server_code
git apply --check /home/deng/Documents/codebase/RLinf/docs/experiments/pi05-explicit-subtask-ce-lora/artifacts/robot_server_compat.patch
```

版本、协议和夹爪单位核实后，由操作人员在 RLinf 目录执行完整启动命令，并额外指定：

```bash
CUDA_VISIBLE_DEVICES=0 bash evaluations/run_eval.sh realworld realworld_spiritai_eval_pi05_subtask_deploy \
  rollout.model.model_path="$PI05_DEPLOY_CKPT" \
  rollout.model.openpi_data.norm_stats_path="$PI05_DEPLOY_CKPT/norm_stats.json" \
  env.eval.override_cfg.robot_url=ws://172.16.0.30:8766 \
  env.eval.override_cfg.execution_enabled=true
```

该命令复用上一节设置的 `PI05_DEPLOY_CKPT`。该 profile 使用手动启动命令作为执行入口，禁用了旧键盘 wrapper；开启后预热结束即开始动作，无需再按 `a`。external following 延迟至第一条有效动作发送前开启。初始化和 reset 只读取当前位置，不执行回零或夹爪复位。

取消会停止发送后续 SDK 指令，底层控制器仍可能保持最后一个目标。软件取消不等价于电机急停；真实 SDK 的阻塞、取消与保持行为仍需在现场验证。

## 控制参数为何这样设置

训练数据为 30 Hz，首版部署保留 `source_hz=30`。OpenPI 参考 profile 的 25 Hz 会改变轨迹持续时间，不能直接替换。若以后采用其他控制频率，应同时处理动作重采样、延迟步数和 model-space RTC prefix 的索引关系。

相机尺寸按当前 robot launch 的 320×240 设置，使用 `cam_high`、`cam_left_wrist`、`cam_right_wrist`，进入模型后仍按既有 transform 处理为 224×224。container 已旋转腕部图像，client 不重复旋转。

动作布局为 `[left_pose6, left_psi, left_gripper, right_pose6, right_psi, right_gripper, torso_pose6, base3]`。发送时去掉两个 psi，按 container 能力选择 14/20/23D；14D 不包含 torso，20D 不包含 base。FoldBox profile 强制 base 为零。

姿态按 container 源码的 rotation vector 语义处理。相对旋转通过 SO(3) 计算并限制旋转角，平移通过向量范数限制速度；夹爪限制单 tick 变化量。新 profile 另外将夹爪 command 限制在示范数据使用的 `[0, 0.1]`，这是工程取值，并非已验证的硬件规格。执行时测量值超出配置范围会报错。现场必须确认 state 与 command 的单位、方向及合法范围。

## 超时与退出

新配置为 bootstrap/warmup 提供 30 秒预算，为运行中的 replan 提供 1.2 秒预算。后者是本机启动值，仍需实测真实网络及最长文本的延迟。动作 chunk 的时间跨度为 `50/30` 秒，重规划在至少执行 10 个模型 tick 后发起；不能靠不断增大 timeout 来弥补持续超出剩余动作时间的推理。

env worker 同时最多保留一个推理请求。接受回复时核对 session、chunk、错误、有限值和 deadline。控制索引按照本地 monotonic clock 推进，RPC 超时耗费的时间也会推进动作位置，禁止用其他主机的原始 monotonic timestamp 计算延迟。新 chunk 跳过推理期间已经过时的前缀。

发生超时、chunk 耗尽、无 EOS、错配回复或执行异常后，本轮停止接受预测动作。已经成功执行过控制 tick 时，尝试两次保持实测位姿，随后取消 dispatch、释放本连接开启的 following 并关闭连接。warmup/bootstrap 失败时不发送 hold 或开启 following。trace 保留 hold 和 cleanup 的成败。错误以 `rtc_aborted=1` 回传，runner 收尾并以非零状态退出；排查完成后重新启动新 session。

本部署使用 RLinf 推理期 `approx` guidance。OpenPI 当前参考入口要求的 `training_time_v1` 属于另一种 RTC 契约，不能通过修改 metadata 名称来兼容当前 checkpoint。

## 验证证据与边界

验证使用一张 NVIDIA RTX PRO 5000 Blackwell、现有 `.venv`、真实 45000-step 权重和原训练 stats。CPU 单测覆盖协议错配、执行错误、观测模式、SO(3) 限幅、session 隔离、wall-clock 索引和动作耗尽。原生入口测试运行真实 Ray、model、worker、WebSocket server 及插值代码，仅将机器人 SDK 和相机替换为本地模拟。

测试命令和结果保存在 [deployment_validation.json](artifacts/deployment_validation.json)，完整日志与临时测试脚本归档在 `logs/inference/20261111_deploy_validation/`。观测模式、模拟执行与注入延迟测试都未连接真实机器人，测试过程中没有训练或保存新权重。

| 检查 | 结果 |
|---|---|
| CPU 回归 | robotics 339、real-env 258、models 77 项通过；1 项可选 Gemma3 import 测试跳过 |
| 单 GPU / batch=1 | 80 个 episode 各选一个不同阶段观测；EOS 100%，动作全部有限，生成 10 种 subtask 文本 |
| 预热后独立 HL＋LL 推理 | P50 0.672 秒、P95 0.854 秒、最大 0.895 秒；关闭 RTC，未包含真实网络 |
| PyTorch 峰值已分配显存 | 6.639 GiB；不等于 NVML 总显存占用 |
| 原生只读观测 | 150 tick，SDK 动作下发为零 |
| 原生模拟执行 | 300 tick，完成后 following 关闭 |
| 运行中注入 1.5 秒延迟 | 超过 1.2 秒预算后 abort，2 次实测 hold，随后关闭连接 |
| 启动超时 | trace 明确出现 warmup 超时；没有 step、hold 或 SDK 动作下发 |

上述延迟与成功退出证明接口可运行，不构成真实控制周期或物理折盒成功率测量。RTC 继续使用 model-space 预测尾部；限幅可能改变实际下发指令，需要结合 trace 审查两者差异。

运行含 Ray worker 的测试文件时，使用独立进程串行执行，并向 worker 导出完整导入路径：`PYTHONPATH="$PWD:$PWD/tests:$PWD/tests/unit_tests"`。并行运行机器人套件会争用同名 actor；仅让 pytest driver 的 `sys.path` 包含测试目录，也不能保证远端 worker 能导入定义在测试模块中的驱动。

保留接口边界和失败原因的断言。`prepare_actions` 返回 NumPy 数组，进入 typed RTC response 检查前需要转回 CPU Tensor。预热失败不能走会启用 following 的 hold 流程。预期失败用例必须检查 trace 中的具体阶段；最初启动超时用例的 10 tick 不满足 chunk 整除约束，改为 150 tick 后才真正验证到 warmup deadline。

离线测试证明接口、数据流和有限时序检查可以运行；它们不能证明真实控制延迟、物理停止能力、规划正确性或折盒成功率。此前离线报告的逐分量 Euler 角误差也不能解释为 rotation vector 的物理角距离；位置误差结论保留，姿态指标应另按 SO(3) 重新计算。
