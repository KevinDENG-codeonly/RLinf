# SpiritAI FoldBox 双卡正式训练预检

2026-10-27（Asia/Shanghai）｜分支：`feature/pi05-flow-only-full`｜基线：`539a1428` + 当前未提交改动。

本页保留 micro batch 1 阶段的预检记录；最新推荐、跨阶段比较边界和可独立复用的命令见[工程复盘总览](README.md)。

## 范围与结论

双卡全参数更新、完整 checkpoint 保存、退出进程后恢复并继续更新，以及 global batch 64/128 的小范围运行均通过。W&B 已使用全局登录在线同步。本次仅做正式实验前的运行验证；正式长训练由人工启动，未连接机器人或下发动作，不评价收敛和任务成功率。

## 设置

- 硬件：2 × NVIDIA RTX PRO 5000 Blackwell，每卡 48,935 MiB。
- 环境：项目 `.venv`，PyTorch `2.11.0+cu130`，W&B SDK `0.25.0`。
- 模型：已转换的 `pi05_base`，3,353,433,872 个参数，LoRA 关闭；FSDP1 `full_shard`，FP32 参数与更新、BF16 计算、FP32 梯度归约，梯度检查点开启。
- 数据：`/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_14Annotations`；使用 `model_checkpoints/foldbox_assets/norm_stats.json`。
- 输入输出：三路图像、Cartesian 25D state token，token 上限 256，action horizon 50、模型维度 32；FoldBox 底盘 action target 置零。
- 每卡 micro batch 1，4 个数据 worker，stateful loader 开启。
- 预检学习率恒定 `1e-5`，warmup 为 0；与正式 recipe 的 warmup 设置不同。所有训练测试均有明确步数上限。

## 实测结果

| 测试 | 更新次数 | 最后一步 loss | 最后一步训练耗时（秒） | GPU 0/1 采样峰值（MiB） | W&B |
|---|---:|---:|---:|---|---|
| save_bs32_m1 | 2 | 0.236387 | 15.224 | 47207 / 45903 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/ww2no8oc) |
| resume_bs32_m1 | 1 | 0.205425 | 32.140 | 47563 / 48243 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/1r1vfas6) |
| profile_bs64_m1 | 2 | 0.246201 | 28.268 | 45230 / 44875 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/rn0nhgt2) |
| profile_bs128_m1 | 2 | 0.230262 | 55.495 | 45692 / 44895 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/v9pyovlq) |

耗时取 RLinf 的 `time/training`，排除 checkpoint 保存；两个更新的测试采用第二步数据，首次更新包含数据 worker 冷启动。恢复测试只有一个更新，因此其耗时包含冷启动，不用于稳态吞吐比较。峰值来自 0.5 秒间隔的 NVML 总显存采样，包含桌面等已有占用，也可能漏掉瞬时峰值。保存与恢复测试覆盖的阶段不同于不保存的 batch 测试，不据此推断 batch 增大后显存降低。

checkpoint 检查确认：667 个 Adam state 中，665 个活跃 state 从 step 2 恢复并推进到 step 3；两个未参与当前 action loss 的 prefix 末层 norm state 保持 step 0。各 rank 数据位置从 32 推进到 48。动作头、视觉 stem 和语言骨干代表性权重的最大更新量约为 `1e-5`。未做与未中断三步训练的完整逐参数数值对比。

普通推理与 RLinf RTC 推理使用 step 3 checkpoint、真实 FoldBox 第 0 帧观测和正式真机模型配置，在单卡上通过。输出分别为 `(1, 50, 25)`，原始模型动作为 `(1, 50, 32)`，均为有限值。模型加载约 23.90 秒，首次普通推理约 6.16 秒（含 fused kernel 编译），随后单次 RTC 调用约 0.252 秒。这里只测了一次 RTC 调用，不构成稳态延迟或 p99 基准，也不证明机器人任务效果。

## 正式预算与手动启动

| Global batch | 梯度累积次数 | 与旧 `32 × 90k` 相同样本曝光量的 steps | 按本次单个稳态步骤粗估的纯训练天数 |
|---:|---:|---:|---:|
| 32 | 16 | 90,000 | 15.9 |
| 64 | 32 | 45,000 | 14.7 |
| 128 | 64 | 22,500 | 14.5 |

这些估算只使用短测的一个稳态步骤，不包含中断、保存和长程波动。增大 global batch 主要来自梯度累积，没有带来成倍的样本吞吐提升，也尚未比较优化质量。正式启动前应先确定可接受的总训练时长。

batch 64 的一次 optimizer 更新包含每卡 `64 / (2 × 1) = 32` 次前向和反向。旧 smoke 的 global batch 为 2，一步仅含一次前向和反向，因此不能直接比较两次实验的“秒/步”。当前约 3.35B 参数全量训练还开启了梯度检查点和 FSDP full shard，分别增加重算及跨卡通信开销；训练未开启 `torch_compile`。尚未采集分项 profiler，不能据此给出计算、通信和数据加载各自的耗时比例。

45,000 步来自旧 LoRA recipe 的 `32 × 90,000` 样本预算换算，合计 2,880,000 个训练窗口，约等于本数据集两遍，并非已经验证的全参数训练收敛要求。5000 步是可恢复的正式首阶段预算；后续总步数应结合曲线与真机验证确定。

已生成当前机器可用的手动启动脚本：`logs/spiritai_preflight_20261027_131340/train_formal_bs64.sh`，尚未执行。它采用 batch 64、45,000 总步数、750 warmup steps、每 2,500 步保存。可用 `MAX_STEPS` 限定首阶段，scheduler 仍保留完整 45,000 步计划；恢复时保持训练数据、归一化、micro batch、worker 数、seed 和布局一致。

后续 PCIe 调优已将本机推荐参数更新为 P2P + micro batch 8，并验证保存和恢复。正式启动应使用 [PCIe 调优报告](2026-10-27-pcie-tuning.md)中的 `logs/spiritai_tuning_20261027/train_formal_bs64_pcie.sh`；下方原始脚本及本报告耗时仍对应 micro batch 1 的预检基线。

```bash
# 首阶段正式实验示例；由人工执行，约需 1.6 天纯训练时间。
MAX_STEPS=5000 bash logs/spiritai_preflight_20261027_131340/train_formal_bs64.sh

# 完整等样本曝光量实验；由人工执行，短测粗估约 15 天。
bash logs/spiritai_preflight_20261027_131340/train_formal_bs64.sh
```

已准备空的 tmux 会话 `spiritai-full`，其中尚未运行训练。先进入会话，再由人工执行上面的首阶段或完整训练命令：

```bash
tmux attach-session -t spiritai-full
```

离开时按 `Ctrl+b`，松开后按 `d`，训练会在会话中继续；重新连接使用相同的 attach 命令。若会话不存在，先用 `tmux new-session -s spiritai-full -c /home/deng/Documents/codebase/RLinf` 创建。tmux 可以保留终端断连后的进程；机器重启或训练进程被终止时，仍需要从 checkpoint 恢复。

每个包含训练状态和完整权重的 checkpoint 约 50 GiB；上述完整预算约产生 18 份，约 900 GiB。当前保留的预检 checkpoint 共约 100 GiB。应结合保留策略和实际可用磁盘空间管理正式产物。

## 环境修正与后续真机工作

启动时修正了两项环境问题：直接运行入口需要导出仓库 `PYTHONPATH`；全局 W&B CLI `0.31.0` 默认服务为 `https://api.forge.coreweave.com`，项目 SDK `0.25.0` 默认连接旧地址。测试通过 `WANDB_BASE_URL` 复用 `.netrc` 中已有登录，没有复制凭据或改动项目依赖。EN/ZH 使用说明已同步更新并构建通过。

真机阶段仍需确认 container 的实际地址、相机分辨率、控制频率及现场起始构型。先做只读 observation/metadata 检查，再在现场验证 hold、ACK 和逐行动作时序。已实现的部署采用 RLinf 推理时 `approx` RTC；训练保持普通 Flow Matching，未迁移旧训练时动作前缀条件。

原始证据在上述日志目录：各测试的 `command.json`、`run.log`、`summary.json`、`gpu_samples.json`，以及汇总 `results.json`、`checkpoint_audit.json`、`inference_audit.json`。首次入口和 W&B 地址匹配失败的日志分别保留在 `logs/spiritai_preflight_20261027_131027/` 与 `logs/spiritai_preflight_20261027_131110/`；修正后测试通过。AI assistant 执行了本报告的小范围预检。正式训练未启动。
