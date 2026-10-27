# SpiritAI 双卡 PCIe 通信调优

2026-10-27（Asia/Shanghai）｜分支：`feature/pi05-flow-only-full`｜基线：`539a1428` + 当前未提交改动。

跨阶段工程结论和独立启动命令见[总览](README.md)；小型 CSV、审计摘要与图表已归档到 [artifacts](artifacts/README.md)，避免本机日志清理后丢失依据。

本报告记录同一真实 FoldBox 数据和 π0.5 全参数 SFT 配置下的 PCIe 通信对照，给出本机正式启动参数。推荐 `NCCL_P2P_LEVEL=PHB`、global batch 64、每卡 micro batch 8、FSDP1 `full_shard`。20 步确认运行排除前 3 步后平均为 **6.695 秒/步**，相对初始 micro 1 基线吞吐约提高 **4.19 倍**；完整 checkpoint 保存和新进程恢复到第 21 步通过。正式长训练仍由人工启动，本次未连接机器人或发送动作。

## 硬件路径与单位

两张 RTX PRO 5000 Blackwell 各有 48,935 MiB 显存。在负载下，两卡均报告 PCIe Gen5 ×16；`nvidia-smi topo -m` 显示同一 NUMA 节点内的 `NODE` 路径，读、写 P2P 能力矩阵均为 `OK`。这条路径跨 PCIe 主桥，GPU 直接 P2P 仍经过主桥，并不等于 NVLink 直连。

初始 RLinf/NCCL 日志的 GPU Channel 使用 `SHM/direct`。仅在启动进程的环境中加入 `NCCL_P2P_LEVEL=PHB` 后，同一训练入口使用 `P2P/IPC`。RLinf NVIDIA worker 默认的 `NCCL_CUMEM_ENABLE=0` 保持一致。日志中的 `Using network Socket` 是初始化/网络后端信息，应以 GPU Channel 的 `via P2P/IPC` 或 `via SHM/direct` 判断本机实际传输路径。

本报告带宽均为 **GB/s（十进制字节/秒）**，不是 Gb/s；26.7 GB/s 约为 214 Gb/s。PCIe Gen5 ×16 的理论单向带宽为 `32 GT/s × 16 × 128/130 ÷ 8 ≈ 63.0 GB/s`，尚未扣除事务层开销。[PCI-SIG 带宽表](https://pcisig.com/sites/default/files/files/PCI-SIG-Seamless_Transition_to_PCIe_5.0_in_System_Implementations_FINAL.pdf)

两个方向分别进行的单向跨 GPU Tensor 拷贝，在 256 MiB 和 1 GiB 消息下实测为 41.3–41.7 GB/s，每组 10 次预热、30 次计时，并同步两卡和检查数据。它与双 rank 的 NCCL 集体通信不是同一负载；这些测试不能独立确定理论值到实测值之间各项开销的占比。

NCCL 表中采用标准 bus bandwidth：两 rank all-gather/reduce-scatter 的 `busbw = algbw × (n-1)/n = algbw/2`。例如 P2P all-gather 的 algbw 约 53.45 GB/s，对应 busbw 约 26.73 GB/s。[NVIDIA NCCL Tests 计算方式](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

| 256 MiB 完整消息 | 默认 SHM | P2P 自动 CTA | P2P 固定 8 CTA | P2P 固定 16 CTA |
|---|---:|---:|---:|---:|
| all_gather | 22.79 | 26.73 | 29.28 | 29.73 |
| reduce_scatter | 23.15 | 26.54 | 28.67 | 29.39 |

单位为 GB/s。通信短测使用两个独立进程，每个进程只暴露一张卡，与 RLinf actor 的隔离方式一致。all-gather 使用 BF16，reduce-scatter 使用 FP32；消息覆盖 8 KiB、256 KiB、4 MiB、32 MiB、256 MiB。每项先预热 15 次，再计时 100 次或 30 次，比较两 rank 中较慢的总耗时，并检查结果值。实际加载的 NCCL 日志版本为 `2.32.3+cuda12.9`。

## 实际训练对照

全部对照从同一 pi05 基础权重启动，global batch 固定为 64，8 个 optimizer updates，排除前 3 步，取后 5 步的 RLinf `time/training` 平均值。时间从在线 W&B 的本地原始 history 提取，包含数据等待、前后向和 optimizer 更新，不包含模型初始化与 checkpoint 保存。显存为每 0.5 秒采样的 NVML 总占用峰值，包含桌面占用，也可能漏掉瞬时峰值。

数据为 `20260805_FoldBox_SpiritAI_Moz1WB_14Annotations`，仍使用原 norm stats、三路图像、25D Cartesian state token、256 token 上限、horizon 50 和底盘 target 置零。FP32 参数与优化器更新、BF16 主干计算、FP32 归约、梯度检查点、4 个数据 worker、stateful loader、seed 42 保持一致；`torch_compile=false`、CPU offload 关闭。学习率为恒定 `1e-5`、warmup 为 0，与正式计划的 warmup 不同。本轮按要求聚焦 PCIe，没有测试算子编译。

| 配置 | 每卡累积次数 | 秒/步 | 相对基线吞吐 | GPU 0/1 峰值（MiB） | W&B |
|---|---:|---:|---:|---|---|
| 默认 SHM，micro 1 | 32 | 28.056 | 1.00× | 46291 / 45917 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/p392axsm) |
| SHM + backward prefetch，micro 1 | 32 | 28.078 | 1.00× | 46900 / 46391 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/fi9p0h7c) |
| P2P，micro 1 | 32 | 24.448 | 1.15× | 46345 / 45949 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/p7jtp3t8) |
| P2P，micro 2 | 16 | 13.514 | 2.08× | 46255 / 45853 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/b4vtccb3) |
| P2P + shard_grad_op，micro 1 | 32 | 15.180 | 1.85× | 48258 / 48327 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/6t40tj9h) |
| P2P，micro 4 | 8 | 8.605 | 3.26× | 46434 / 45873 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/4ccco9gn) |
| P2P，micro 8 | 4 | 6.526 | 4.30× | 46698 / 45625 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/p92u8j64) |
| P2P + 16 CTA，micro 8 | 4 | 6.501 | 4.32× | 47861 / 46767 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/h33va7eg) |
| P2P，micro 32 | 1 | 5.846 | 4.80× | 47943 / 48151 | [run](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/mjtio3s9) |

仅切换 P2P、仅开启 backward prefetch，以及同 micro 1 下改用 `shard_grad_op`，这 8 步记录的 loss 和梯度范数均与基线一致。CTA 16 与默认 CTA 的 micro 8 记录也一致。不同 micro batch 会改变随机时间、噪声和增广的抽样分组，因此不同 micro batch 之间不要求逐步 loss 相同；所有已完成运行的 loss 和梯度范数均为有限值。这些短测不构成收敛或任务成功率对比。

选择 micro 8 的依据是速度和显存余量：micro 32 的短测平均为 5.846 秒，但采样峰值达到 48,151 MiB，仅余约 784 MiB；`shard_grad_op` 的 micro 1 峰值为 48,327 MiB。它们未做保存/恢复测试。micro 8 的 8 步平均为 6.526 秒，峰值为 46,698 MiB。固定 16 CTA 只带来约 0.4% 的训练耗时差，却增加 1163 MiB（约 1.14 GiB）峰值显存，因此使用 NCCL 默认 CTA 选择。

P2P 路径本身将 micro 1 耗时从 28.056 秒降为 24.448 秒，减少约 12.9%。增大 micro batch 同时减少完整前后向轮数、重复参数 all-gather 和 kernel/Python 调度，并提高矩阵计算规模；没有分项 profiler，不能把整个组合收益都归因于 PCIe 带宽。框架的 `no_sync` 延迟梯度同步已开启，它与参数 all-gather 是不同的通信。

## 保存与恢复

选中的 micro 8 配置另外运行 20 步并保存完整 checkpoint，随后启动新进程，从 `global_step_20` 恢复并只更新到 `global_step_21`，再次保存。确认运行的后 17 步为 6.418–7.081 秒，平均 6.695 秒；保存这一步的总耗时不作为训练吞吐。确认运行的采样显存峰值为 46626/45863 MiB，恢复运行为 48130/47989 MiB。恢复后的首步含冷启动，不用于稳态性能比较。

checkpoint 审计确认 667 个 optimizer state 中的 665 个活跃 state 从 step 20 推进到 21；两个未参与当前 loss 的 prefix 末层 norm state 保持 step 0。每个 rank 已消费的 micro batch 计数从 80 推进到 84；RNG checkpoint 存在。导出的 667 个参数 Tensor 合计 3,353,433,872 个元素；动作头、视觉 stem 和语言骨干的代表性权重在恢复后均继续更新。未做与不中断 21 步运行的逐参数等价比较。在线记录：[20 步确认](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/i5opj3mi)、[恢复更新](https://app.forge.coreweave.com/zhipeng-deng-rwth-aachen-university/rlinf/runs/imck5q48)。

## 主机 RAM 与 swap 复核

正式启动前另行复核了主机内存。机器约有 125 GiB RAM，空闲时 `MemAvailable` 约 111 GiB；现有 2 GiB swap 已使用约 1.99 GiB，但 `vmstat 1 5` 的后续采样均未发生换入或换出。已占用 swap 不代表当前 RAM 已不足，判断余量应参考包含可回收缓存的 `MemAvailable`。[Linux 内核定义](https://docs.kernel.org/filesystems/proc.html#meminfo)

从上述两次 W&B 原始系统采样中，20 步确认运行共取到 14 个可用 RAM 样本，最低约 47.32 GiB；恢复运行共取到 8 个样本，最低约 55.16 GiB。这些是间隔采样，不能排除更短的瞬时峰值，但保存和恢复已在原有 2 GiB swap 下实际通过。当前终端及各级父 cgroup 的 `memory.max`、`memory.high`、`memory.swap.max` 均为 `max`，未发现这些显式限制。

RLinf 保存时会将分片训练状态暂存到 CPU，并在 rank 0 导出完整权重，确实会增加主机 RAM 占用。约 50 GiB 的 checkpoint 包括约 12.5 GiB FP32 完整权重和训练状态，磁盘大小不等于同等额外 RAM 需求。在模型和优化器配置保持一致时，这些核心状态的尺寸基本固定，不会随 optimizer step 线性增长；多份 checkpoint 的磁盘占用会累积。

按当前要求，swap 总量为约 32 GiB：保留原 2 GiB，新增 30 GiB，作为主机瞬时压力的备用空间；这不是训练配置已测得的运行必需条件。脚本准备时尚未执行，后续文档归档已观察到两份 swap 启用且 `/etc/fstab` 持久化条目存在。脚本另存于 [artifacts/increase_swap_to_32g.sh](artifacts/increase_swap_to_32g.sh)，只适用于同样的初始 2 GiB 状态；当前本机无需重复执行。持续换页会影响训练与保存速度，swap 也不能解决 CUDA 显存 OOM。测试时的证据在 `host_memory_wandb_audit.json`、`host_memory_current.json` 和 `host_cgroup_audit.json`，后续状态见[归档清单](artifacts/experiment_manifest.json)。

### 同一进程连续保存复核

为检查保存后是否累积保留大块内存，另用同一配置运行 7 个 optimizer steps，每 2 步保存。RLinf 在最终步也保存，因此实际完成第 2、4、6、7 步共 4 次完整 checkpoint 写入。每约 0.5 秒采样 `/proc/meminfo` 和本终端 cgroup 的匿名内存、文件缓存，共 578 个样本。

| 保存步数 | 保存窗口可用 RAM 最低采样值（GiB） | cgroup 匿名内存峰值（GiB） |
|---:|---:|---:|
| 2 | 54.81 | 56.30 |
| 4 | 53.79 | 57.19 |
| 6 | 53.07 | 58.05 |
| 7 | 52.53 | 58.42 |

4 次保存均完成，完整权重均可打开，参数数量均为 3,353,433,872，DCP metadata 和数据计数匹配各保存步数。保存结束后匿名内存明显回落，进程退出后系统可用 RAM 回到约 111 GiB；未复现每次保存后保留整份 checkpoint 并最终耗尽 RAM 的问题。匿名内存峰值存在约 2.1 GiB 的增长，短测不能证明多天运行完全没有增长。证据保存在 `host_memory_three_saves/host_memory.jsonl` 和 `repeated_save_memory_audit.json`。

这次专项测试新生成的 4 份临时 checkpoint 已在审计后清理，释放约 199.92 GiB 磁盘空间；日志、文件清单、内存记录和审计 JSON 保留。清理前后 `MemAvailable` 约为 111.83/111.94 GiB，基本不变，而文件缓存从约 65 GiB 降到约 15 GiB。这说明这些文件缓存原本就是可回收内存，不能据其大小判断 RAM 被永久占用。清理记录为 `temporary_checkpoint_cleanup.json`。

## 手动正式启动

本机的启动脚本复用已有任务脚本，仅增加 `NCCL_P2P_LEVEL=PHB` 和 `actor.micro_batch_size=8` 两项覆盖。PHB 是针对本机拓扑实测的进程环境设置，不写入通用 SpiritAI YAML 或系统环境；通用 YAML 保留保守基线。所有 case 保持全参数 FP32 更新。脚本已通过 `bash -n` 和 Hydra `--cfg job --resolve` 检查；配置解析不会启动训练。

进入已有的 tmux 会话，再手动启动正式首阶段：

```bash
tmux attach-session -t spiritai-full

# 在 tmux 内执行；MAX_STEPS 限制本阶段，scheduler 仍按 45000 步规划。
MAX_STEPS=5000 bash logs/spiritai_tuning_20261027/train_formal_bs64_pcie.sh
```

离开时按 `Ctrl+b`，松开后按 `d`；重新连接仍使用上述 attach 命令。首阶段粗估约 9.3 小时纯训练时间。完整 45000 步与旧 `32 × 90000` 样本预算相同，按确认短测粗估约 3.5 天；这些估算不含保存、中断和长程波动，也不是已验证的收敛要求。正式脚本使用 750 步 warmup、每 2500 步保存和已有全局登录的 W&B。

从正式 checkpoint 续训时，保持数据、stats、micro batch、worker 数、seed 和 rank 布局一致，使用完整 45000 步计划：

```bash
bash logs/spiritai_tuning_20261027/train_formal_bs64_pcie.sh \
  +runner.resume_dir=/path/to/checkpoints/global_step_5000
```

本次两个确认 checkpoint 约占 100 GiB，与较早预检的约 100 GiB 一起保留。完整正式预算约产生 18 份 checkpoint、约 900 GiB，需按实际空间管理保留数量。

## 证据与边界

原始证据保存在 `logs/spiritai_tuning_20261027/`：各 case 的 `command.json`、`run.log`、`history.json`、`summary.json`、`gpu_samples.json`，NCCL 路径日志、通信微基准 JSON、`checkpoint_audit.json`、`formal_resolved.yaml` 和总表 `results.json`。脚本副本保存在同目录的 `reproduce/`。本轮没有新增性能相关的 RLinf 核心代码。`nsys` 未安装，本轮使用 NCCL 路径日志、通信微基准和 native SFT 对照；结论限于当前硬件与有限步数。

相关上游检索发现 Ascend 优化 PR #1622 和 diffusion 传输 PR #1721，均未整套迁入。AI assistant 执行了本报告的有限测试；正式长训练未启动。
