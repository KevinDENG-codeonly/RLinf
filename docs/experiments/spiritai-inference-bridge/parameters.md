# Full Flow 推理参数说明

根据操作目的调整 Full Flow 的运行长度、动作开关、RTC、采样或限幅，并在修改前确认参数来自哪层配置。本文解释当前实现的语义，不修改项目 YAML。[返回操作手册](README.md) · [English](parameters.en.md)

## 参数从哪里来

`key=value` 是 Hydra 启动配置覆盖，不是模型输入，也不是必填的额外 API 参数。所选 primary 的 defaults 先加载，`_self_` 应用本文件设置，命令行最后覆盖同名 key。下列路径均相对于仓库根目录。

| 文件 | 管理的设置 |
|---|---|
| [`realworld_spiritai_eval_pi05_full_deploy.yaml`](../../../evaluations/realworld/realworld_spiritai_eval_pi05_full_deploy.yaml) | Full Flow 的 primary；配置名、模型模式和可持久化的本实验覆盖 |
| [`_spiritai_pi05_deploy.yaml`](../../../evaluations/realworld/_spiritai_pi05_deploy.yaml) | 公共部署参数：只读默认、超时、三图、BF16、采样、限幅覆盖 |
| [`_spiritai_pi05_rtc.yaml`](../../../evaluations/realworld/_spiritai_pi05_rtc.yaml) | 原生 RTC/worker 拓扑、重规划设置、W&B 默认 |
| [`realworld_spiritai_cartesian.yaml`](../../../examples/embodiment/config/env/realworld_spiritai_cartesian.yaml) | 环境模板；默认运行长度、地址、相机、following 设置 |
| [`SpiritaiCartesianConfig`](../../../rlinf/envs/real/spiritai/cartesian.py) | 未在 YAML 中覆盖的平移、旋转和夹爪速度等类默认值 |

日常单次测试优先用 CLI 覆盖，启动日志会保存解析后的完整配置。要长期改变 Full Flow 设置，放入 primary；修改公共片段会影响继承它的配置。primary 若显式写了某字段，后续再改下层模板不会覆盖它。当前操作命令显式传模型和 bundle 路径，因 primary 中这些值仍是占位路径。

## 三个常用参数为什么同时出现

先区分“外层循环最多执行多少步”和“环境何时结束一个 episode”，再决定是否把二者设成同一个值。当前 bounded RTC 每次 `_evaluate_rtc_action()` 对应一行目标/一次环境 step，而不是执行完整 50 行 chunk。

| 参数 | 默认值与来源 | 实际作用 |
|---|---|---|
| `env.eval.max_steps_per_rollout_epoch` | `3000`，环境模板 | `RTCEnvWorker._evaluate_bounded` 的 `for` 循环上限；限制本轮控制 tick 预算 |
| `env.eval.max_episode_steps` | `3000`，环境模板 | `RealEnv.step` 将累计 step 与上限比较，产生 timeout truncation，进而成为 done |
| `env.eval.override_cfg.execution_enabled` | `false`，公共部署片段 | false 仅预测/读取；true 才发送动作并按配置管理 following |

worker 遇到环境 done 会提前 break，也会在循环预算耗尽时结束。当前单环境、`rollout_epoch=1`、`auto_reset=false`，且没有更早的终止/异常时，实际 tick 数通常由两个上限中较小的值决定。

| 外层预算 | episode 上限 | 正常结束位置 |
|---:|---:|---|
| 1500 | 1500 | 1500 tick，预算与环境截断对齐 |
| 1500 | 3000 | 1500 tick，外层预算耗尽，环境可能尚未 done |
| 3000 | 1500 | 1500 tick，episode 截断导致提前 break |
| 1500 | 500 | 500 tick，先到 episode 上限 |

二者相等是操作约定，不是接口强制要求。当前通用 config validator 要求 **`max_steps_per_rollout_epoch` 能被 `num_action_chunks=50` 整除**；推荐长度使用 50、300、1500、3000。`max_episode_steps` 没有同一处整除检查，一起设成相同的正整数是为了保持运行边界清晰。

`execution_enabled=false` 也会计入控制步并运行模型，preview 不是“只加载不推理”。true 时预热与 bootstrap 完成就开始提交动作；它不表示自动回初始位姿，也不包含键盘确认。类本身默认 execution=true，但本部署 YAML 明确覆盖为 false；使用本页指定的 deployment primary 才能得到这里的默认行为。

## 是否可以不在命令中重复输入

可以。保留默认 `3000 / 3000 / false` 时，省略三项即可。若希望固定为 1500 tick 预览，可以把下面片段合并到 Full Flow primary 中；这只是文档示例，本次没有修改该配置：

```yaml
env:
  eval:
    max_steps_per_rollout_epoch: 1500
    max_episode_steps: 1500
    override_cfg:
      execution_enabled: false
```

之后只需在需要动作的那次命令中追加 `env.eval.override_cfg.execution_enabled=true`。也可以在 YAML 中长期设 true，但该 profile 每次启动都会在预热后直接执行；当前保留 false，便于区分预览和动作命令。

修改 primary 中的 `rollout.model.model_path`、`rollout.model.openpi.deployment_metadata_path` 和 `rollout.model.openpi_data.norm_stats_path` 后，也可省略三项路径覆盖。示例中的文件夹必须实际存在且对应同一份权重。模型输入仍由图像、state 和 `task_description` 构成，以上设置不进入 prompt。

## 运行长度与频率

先用实际控制频率换算动作时长，再选最接近且满足整除约束的步数。`source_hz` 是动作时间轴与限幅计算的标称频率，不是严格的闭环吞吐保证。

| 参数 | 默认值 | 作用和调整边界 |
|---|---:|---|
| `env.eval.rollout_epoch` | 1 | 评估轮数；增加会重复 reset/bootstrap，不能当作一条连续动作轨迹延长 |
| `env.eval.auto_reset` | false | episode 边界交给当前 worker；真机部署保留 false |
| `env.eval.override_cfg.source_hz` | 30 | 动作索引、重叠延迟和单 tick 限幅的时间基准；与训练数据一致，不能只为“多跑一会”而改它 |
| `env.eval.override_cfg.robot_url` | `ws://172.16.0.30:8766` | 实际连接的机器人地址；server 监听的 `0.0.0.0` 不是 client 目标地址 |
| `cluster.component_placement.rollout` | primary 继承 0；本机命令覆盖为 1 | 指定唯一 policy GPU hardware rank；不是使用两张卡 |
| `cluster.component_placement.env` | 0 | env 的节点放置，当前为工作站 CPU 侧；不表示再加载一份 GPU 模型 |

动作阶段约为 `N / f_measured`。2026-11-19 的 1500 tick 真机运行首末 step 间隔为 58.97 秒，平均 25.42 Hz；按标称 30 Hz 计算则是约 50 秒。要接近一分钟，用约 `60 × 25 ≈ 1500` tick。初始化、13 GB 权重 hash、加载、编译预热均另计；实际吞吐改变后需重新换算。当前没有文档所述的“严格墙钟 60 秒”参数，不应把 `max_episode_steps=60` 当成 60 秒。

## RTC 与 Flow 采样

分开控制异步调度和模型 guidance，才能只改变想研究的机制。Full Flow 使用当前已验证的 `approx` sampler；部署不会为 checkpoint 增加新的训练目标。

| 参数 | 当前默认值 | 作用和修改说明 |
|---|---|---|
| `runner.rtc.enabled` | true | 选择原生异步 RTC env/rollout workers；正常部署保持 true |
| `rollout.model.openpi.rtc_enabled` | true | 是否在已有 chunk 的条件下使用 sampler guidance；false 仍可保留异步控制器，用于单独对比 guidance |
| `rollout.model.openpi.rtc_guidance_mode` | `approx` | 当前支持的部署 guidance；不要把旧 OpenPI policy-server 的 `training_time_v1` 声明当成同一个参数 |
| `rollout.model.openpi.rtc_guidance_clip` | 5.0 | guidance 随 Flow 时间变化的缩放系数上限，不是 Cartesian 速度上限 |
| `runner.rtc.min_exec_horizon` | 10 | 时间轴动作索引至少推进到 10 且无在途请求时请求下一 chunk；实际请求频率还受推理/RPC 延迟影响 |
| `runner.rtc.initial_delay_steps` | 15 | 初始延迟估计，30 Hz 下为 0.5 秒，用于 guidance 重叠；不是启动等待 0.5 秒 |
| `runner.rtc.delay_buffer_size` | 8 | 保存近期 replan 请求延迟，使用窗口内最大值；初始估计随窗口逐步淘汰 |
| `runner.rtc.request_timeout_s` | 1.2 | replan 的本地墙钟判定预算；超时拒绝迟到动作，并非硬实时抢占所有阻塞 RPC |
| `runner.rtc.bootstrap_timeout_s` | 30.0 | 每个 warmup/bootstrap 请求的预算；不包含此前权重加载 |
| `runner.rtc.warmup_requests` | 2 | 初始 warmup 请求数，每次覆盖普通 Flow 与 RTC 路径；完成后重置并 bootstrap |
| `runner.rtc.shutdown_timeout_s` | 15.0 | driver 等待 worker 收尾的时间预算 |
| `runner.rtc.stop_hold_steps` | 2 | 完成过控制 tick 后异常时，最多尝试两次实测位姿 hold；不是正常结束额外运动两步，也不保证失败链路能完成 hold |
| `env.eval.override_cfg.timeout_s` | 2.0 | 单次机器人连接/RPC 及 idle 等待的超时；与模型重规划预算分别计时 |
| `runner.rtc.fixed_delay_steps` / `runner.rtc.chunk_pause_seconds` | 0 / 0.0 | 仿真用设置；真机路径要求保持为零 |
| `runner.rtc.inject_delay_ms` | 0 | 人工延迟故障注入；日常真机运行保持零 |
| `rollout.model.num_steps` / `rollout.model.openpi.num_steps` | 10 / 10 | Flow Euler 采样迭代数；当前 factory 优先读取后者。调整时同时覆盖二者；变少减少前向次数，动作质量需重验 |
| `rollout.model.openpi.torch_compile` | true | 使用既有局部 Torch 编译；影响首次编译/预热与稳态开销，不是重新训练或整模型构建 |
| `runner.rtc.require_subtask_eos` | true，继承公共片段 | 当前 Full Flow hierarchy=false，不执行 Subtask 生成，因此无须调整该字段 |

50 行 horizon 在 30 Hz 动作时间轴上约覆盖 1.67 秒；RTC 提前请求替换，避免等待全部 50 行再推理。响应到达时根据本地 monotonic time 跳过过期前缀；动作耗尽会中止，不能无限重复最后目标。有效新响应必须在请求预算内返回，并保有可用的动作时间。

## 动作模式与限幅

执行开关决定是否提交指令，限幅决定提交前如何约束目标。以下速度值来自 `SpiritaiCartesianConfig`，公共部署片段覆盖了 gripper/base 部分值。

| 参数（前缀均为 `env.eval.override_cfg.`） | 部署默认值 | 作用 |
|---|---:|---|
| `enable_external_following` | true | 执行模式在第一条有效动作前请求 following；preview 不改变模式 |
| `defer_external_following` | true | 将开启 following 延后到预热/初始推理之后 |
| `require_execution_capabilities` | true | 执行前检查当前适配协议；不能只加三个 true 或关掉检查来代替实现 |
| `min_cartesian_dim` | 20 | 保留 torso，拒绝只支持 14D 的 endpoint；23D 还带零 base |
| `max_arm_translation_m_s` | 0.08 | 左右臂目标相对测量位置的每 tick 平移限幅，单位 m/s |
| `max_arm_rotation_rad_s` | 0.35 | 左右臂 SO(3) 旋转角速度上限，单位 rad/s |
| `max_torso_translation_m_s` | 0.04 | torso 平移速度上限，单位 m/s |
| `max_torso_rotation_rad_s` | 0.2 | torso SO(3) 旋转角速度上限，单位 rad/s |
| `max_gripper_velocity_s` | 0.8 | 夹爪每秒指令变化上限，按示范/SDK 的指令量纲解释，不是未经验证的 m/s 声明 |
| `gripper_min` / `gripper_max` | 0.0 / 0.1 | 夹爪目标范围；与本 checkpoint 的 action stats 约 0～0.1 一致，不能盲目套用旧脚本 reset=1.0 |
| `zero_base_actions` / `max_base_speed` | true / 0.0 | 此折盒实验固定 base 指令为零 |

限幅基于测量状态，平移和旋转约束按 `速度 / source_hz` 计算；当前没有把 `source_hz` 自动替换成实测频率。rotation vector 使用 SO(3) 差值，不按三个 Euler 分量直接裁剪。软件取消停止后续 SDK 提交，不撤销已进入底层队列的目标；following 的开启状态也不代表已经发送了动作。

## 模型契约、任务文本与日志

以下设置关系到训练数据解释或资产身份。保持原值，使用部署校验发现不匹配；它们不适合当作运行时长或速度旋钮。

| 参数 | 当前值 / 用途 |
|---|---|
| `rollout.model.openpi.policy_kind` / `rollout.model.openpi.hierarchical.enabled` | `flow_only` / false；overall 直接条件化动作 |
| `rollout.model.openpi.config_name` | `pi05_spiritai_cartesian`；选择对应输入/输出 transform |
| `rollout.model.num_action_chunks` / `rollout.model.openpi.action_chunk` | 50 / 50；模型 horizon 与输出 chunk，与整轮执行步数不同 |
| `rollout.model.action_dim` / `rollout.model.openpi.action_env_dim` / `rollout.model.openpi.model_action_dim` | 25 / 25 / 32；policy/state 与 model-space 维度 |
| `rollout.model.openpi.num_images_in_input` | 3；high、left wrist、right wrist，输入 RGB 240×320，模型 transform 到 224×224 |
| `rollout.model.openpi.discrete_state_input` / `rollout.model.openpi.max_token_len` | true / 256；与训练的 state token 和文本长度契约一致 |
| `rollout.model.precision` / `rollout.model.openpi.embed_dtype` | bf16 / bf16，保留既有 FP32 islands |
| `rollout.model.is_lora` | false；加载完整 native 权重 |
| `rollout.model.openpi.validate_deployment` | true；核对模式、geometry、stats 与完整权重 SHA256 |
| `rollout.model.openpi_data.norm_stats_path` / `rollout.model.openpi_data.zero_base_action_targets` | 已准备 bundle 的 stats / true；使用同一训练解释 |
| `env.eval.override_cfg.image_height` / `env.eval.override_cfg.image_width` | 240 / 320；必须匹配 container 的实际输出 |
| `env.eval.override_cfg.task_description` | 模型的 overall 指令；Full Flow 不手工拼接 current subtask |
| `env.eval.keyboard_reward_wrapper` | null；本配置没有按 a 才执行的键盘 gate |
| `env.eval.runtime_setup` | false；使用外部 SDK container，保留本机 ROS 进程 |
| `runner.logger.logger_backends` | YAML 为 `[wandb]`；本机操作命令覆盖为 `[]`，保留终端/RTC trace |
| `runner.logger.project_name` / `runner.logger.experiment_name` | `rlinf` / `spiritai_pi05_full_deploy`；云端命名与 trace 子目录 |
| `runner.logger.log_path` | 启动脚本自动生成时间戳目录；一般接受脚本默认 |

要覆盖包含空格的 overall，给整个 Hydra 参数加 shell 引号，例如：

```bash
'env.eval.override_cfg.task_description="Fold the cardboard box and place it on the right side of the table."'
```

通用模型模板中还会打印 `noise_method`、`train_expert_only`、`lora_rank` 等字段；本配置 `task=eval / is_lora=false`，不能把这些训练或其他 task 的字段当作本路径已生效的推理旋钮。

这是供添加到启动命令的参数片段，不是单独可执行的 shell 命令。完整命令、W&B endpoint 和日志诊断见[操作手册](README.md)。
