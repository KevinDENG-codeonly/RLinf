# Explicit Subtask LoRA 真机推理阶段总结

本轮实验于 2026-11-24 暂告一段落。45000 步 LoRA 权重已在单 GPU 上打通显式 subtask、Flow Matching、RTC 和真机动作下发。5 步动作采样通过了 300 tick 真机短测。1500 tick 长测在 1452 tick 后中止，现场报告夹爪过度接触桌面、关节过载并触发急停，同时观察到动作比此前使用的 training-time RTC 模型卡顿。闭环阶段切换、完整折盒与运动平顺性仍未通过验证。[English](lora-validation-2026-11-24.en.md)

## 恢复模型和离线评估

部署入口使用 `realworld_spiritai_eval_pi05_subtask_deploy`。在 rollout worker 加载已合并的 LoRA 权重，`is_lora=false` 表示无需在加载时挂接 adapter。工作站一张 GPU 同时完成高层生成和低层采样，机器人继续使用已经部署的 `robot-server/0.3.1-rlinf-compat`。权重、归一化和部署清单由 loader 校验。

| 资产 | 本机路径 |
|---|---|
| 45000 步合并权重 | `logs/20261102_233335-pi05_subtask_lora_bs64/offline_eval_20261110_163331/exports/step_45000/` |
| 部署清单与训练归一化 | `model_checkpoints/spiritai_deploy/explicit_subtask/{openpi_deployment.json,norm_stats.json}` |
| LeRobot v2.1 测试集 | `/home/deng/Documents/dataset/20261109_FoldNJBox_Moz1_subtask_lora-test/` |
| 逐帧预测与汇总 | 训练 run 下的 `subtask_eval_20261123_095439_1daxGq/results/` |

离线入口为 `.venv/bin/python -m toolkits.standalone_eval_scripts.openpi.spiritai_subtask_eval`，运行时传入 `--dataset-root`、`--checkpoint`、`--deployment-metadata`、`--norm-stats` 和新的 `--output`。使用 `--phase-map` 可按已有标签映射计算阶段匹配率，报告中的 `phase_map.json` 保存了本轮使用的映射。模型只接收三路图像、状态和 overall 指令，参考 subtask 用于选择评估帧和生成后的评分。

80 个 episode 共抽样 1516 帧，全部生成非空文本并输出 EOS。阶段中部匹配为 791/798（99.1%），边界窗口匹配为 523/720（72.6%）；标注切换前后约 0.1 秒的匹配率分别为 68.8% 和 55.0%。这些是录制观测下的当前阶段预测结果。闭环规划还要检验模型能否执行子任务并在任务状态改变后选择下一步。

## 已验证的真机配置

保持 50 步 action chunk、标称 30 Hz 和 `planning_interval=3`。在 LoRA 部署命令中同时加 `rollout.model.num_steps=5` 和 `rollout.model.openpi.num_steps=5`，使动作采样使用 5 次 Euler 积分。打开 `runner.rtc.enabled=true` 和 `rollout.model.openpi.rtc_enabled=true`，采样 guidance 用 `approx`，保留 `runner.rtc.request_timeout_s=1.2`。只读检查显式设定 `env.eval.override_cfg.execution_enabled=false`；[公共启动参数](parameters.md)列出各字段的作用。现场测试关闭了可选 W&B logger。

| 运行 | tick 完成 / 上限 | 最慢高层刷新请求 | 结果 |
|---|---:|---:|---|
| 10 步采样真机短测 | 50 / 50 | 无已接受的刷新 replan | 动作下发通过 |
| 10 步采样真机执行 | 253 / 300 | 已接受请求 1.154 秒，第 17 个请求超时 | 两次 hold，正常关闭 |
| 5 步采样真机执行 | 300 / 300 | 0.735 秒 | 11.64 秒动作阶段，无 abort |
| 5 步采样只读预览与姿态对照 | 300 / 300，另有三轮 50 / 50 | 0.751 秒 | 高层正常生成，始终返回「推盒坯并摆正」 |
| 5 步采样闭环长测 | 1452 / 1500 | 0.756 秒 | 55.95 秒动作阶段，现场报告急停；软件记录 SIGINT |

短测支持保留 5 步采样作为已验证的时序配置。两轮真机测试的高层耗时也有变化，因此时延差不能全归因于低层积分步数。40 个录制样本的配对检查固定输入、subtask 和噪声，5 步与 10 步的前 10 行平均手臂位置误差为 17.0 和 18.1 mm，50 行平均旋转误差为 10.09° 和 9.68°。该检查用于判断是否值得短测，物理轨迹质量仍以现场与控制器数据为准。八轮 trace 的路径、哈希与统计已归档。[真机证据](artifacts/realworld_explicit_subtask_20261124.json)

## 长测事故与阶段判断

长测有 40 次高层生成被接收（包含 bootstrap），全部输出相同的摆正指令，未出现自主 subtask 切换。此前手动摆正盒坯并调整机械臂的只读试验也没有切换。录制图像与真机当前位置的离线交叉试验中，2 个边界帧换用真机姿态后都退回摆正，24 个阶段中部帧则保持正确。边界样本有限，图像中也含机械臂；目前只能确认部分接近边界的输出对姿态敏感。

现场报告夹爪过度接触桌面、关节过载和硬件急停。RLinf trace 的终止原因为 `InterruptedError: RTC driver requested stop: SIGINT`，随后记录两次 hold 与 `closed`。硬件侧没有同步的关节力矩、接触力或保护状态日志，软件收尾记录无法证明机械停止或恢复。当前按测量状态对每行目标做速度和夹爪限幅，验证范围不包含台面碰撞或接触力保护。实验目前暂停在此处，恢复真机执行前需要先核实桌面间隙、接触风险和保护状态。

## 动作卡顿的现有证据

操作人员认为这次动作比此前的 training-time RTC 模型更卡顿。长测 env step 的间隔中位数为 37.88 ms，P99 为 49.28 ms，最大 53.77 ms。最慢的规划刷新请求为 0.756 秒，控制循环在等待推理时仍消费旧块。**请求耗时和控制步间隔是不同量**。该日志没有看到每次高层推理都让运动停约 0.75 秒的现象；25.93 Hz 的平均频率也无法证明动作平顺。

按时间推进动作索引后，在同一个 chunk 内有 230 次相邻控制 tick 跳过了一行模型动作。切块处左腕相邻选中目标的平移差中位数为 536 mm，限幅后的相邻命令差中位数为 1.32 mm，同块内对应值为 260 mm 和 1.10 mm。这些统计描述 env 选中的目标与限幅结果，控制器的速度、加速度和关节 jerk 需要独立采集。目标在切块处变化较大并经限幅，提供了一个值得核查的方向。

当前 RLinf 配置使用推理期 `approx` guidance，在采样中引导动作贴近上一块的剩余目标。每个 env tick 基于实测状态限幅，然后发送单行目标、等待 SDK 下发完成、再读取下一次观测。本地 OpenPI 参考分支 `origin/rct-rebuild@086bfa1` 的 `training_time_v1` policy 使用固定动作前缀。其 `examples/spirit-ai/main.py` 也逐行发送，但依据已接受的命令历史做过渡混合等过滤，并在这一发送点等待 ACK。参考代码同时改变了采样契约和执行方式，不能只用 training-time RTC 解释现场差异。现场差异可能来自模型动作连续性、切块与跳行、串行 RPC 或控制器跟随，现有记录没有在相同盒型、起始姿态和执行参数下做配对，无法分离各因素。若恢复实验，优先同步记录选中目标、下发目标、机械臂实测轨迹、力矩与保护事件，再比较两种 RTC policy。阶段识别与接触风险单独验收。
