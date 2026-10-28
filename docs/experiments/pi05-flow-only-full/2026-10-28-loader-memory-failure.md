# 正式训练第 506 步退出：DataLoader 主机内存增长

2026-10-27 启动的 5000 步正式首阶段在完成 506 次更新后，于 18:27:40 触发 Ray 的主机内存保护。停止原因是数据加载子进程持续占用 RAM，达到 Ray 默认 95% 阈值。需要先解决数据读取内存增长，再恢复正式长训练的准备状态。

## 运行配置与直接证据

运行目录为 `logs/20261027_172946-spiritai_foldbox_pi05_full_bs64/`，W&B run 为 `wllwedmv`。实际配置是 global batch 64、每卡 micro batch 8、两个 actor rank、每 rank 4 个 DataLoader worker，共 8 个数据子进程。`runner.max_steps=5000`、scheduler 总步数 45000、warmup 750、保存间隔 2500；配置没有在 500 步停止的限制。

训练完成 506 次更新，W&B 最后一个零起始 history step 为 505。退出前 loss 为 0.035263、grad norm 为 0.218734，每步训练约 6.39 秒，均为有限值。日志明确记录：

```text
2 worker(s) were killed due to the node running low on memory.
Memory on the node ... was 119.08GB used / 125.24GB limit (0.950782)
Ray killed 2 worker(s) based on the killing policy
```

Ray 的这段输出以 `GB` 标记二进制容量；约为 119.08 / 125.24 GiB。`raylet.out` 记录的时间为 2026-10-27 18:27:40，随后向两个 actor 进程组发送 SIGTERM。Ray object store 使用量为 0。这次终止发生在普通训练更新期间，尚未进入第 2500 步的 checkpoint 保存。

## 内存增长发生在哪里

本地 W&B 二进制记录提取出 225 个主机内存样本。可用 RAM 持续下降：

| 本地时间 | 可用 RAM（GiB） | 主机内存占用（%） |
|---|---:|---:|
| 17:31:29 | 81.02 | 35.31 |
| 17:41:29 | 67.84 | 45.84 |
| 17:51:29 | 60.88 | 51.39 |
| 18:01:29 | 46.76 | 62.67 |
| 18:11:29 | 28.01 | 77.63 |
| 18:21:29 | 14.38 | 88.52 |
| 18:27:29 | 6.51 | 94.80 |

Ray 的终止快照中，8 个 `multiprocessing.spawn` 数据子进程分别占用约 19.09、14.61、12.53、12.31、10.35、10.23、9.24、9.22 GiB，合计 97.58 GiB。两个训练 actor 本身约为 2.80 / 2.83 GiB。结合父子进程记录，主机内存压力集中在数据子进程。

扩大 swap 不会自动扩大 Ray 的物理 RAM 阈值。不能用提高阈值或关闭 Ray memory monitor 代替排查持续增长；那只会把终止推迟到更高内存压力。进程退出后主机可用 RAM 恢复至约 114 GiB，也符合这次占用属于运行中的进程，而非需要删除旧权重才能释放的磁盘缓存。

## CPU 隔离与证据边界

诊断只读取真实数据，不加载模型。单个 stateful DataLoader worker 在 75.6 秒内读取 90 个 micro batch、共 720 个样本；RSS 从第 10 个 batch 的 2136 MiB 升至第 90 个的 6022 MiB，USS 从 1483 MiB 升至 5352 MiB。已经可以在训练计算之外复现增长。

相同配置直接逐条读取数据、尚未经过批量 collate/多进程传输时，400 个样本的进程 RSS 从约 2710 MiB 升至 2935 MiB；该测试达到 100 秒上限后主动结束。单个视频反复解码 300 次，RSS 约从 893 MiB 升至 1002 MiB。两组都不是正式多进程路径的等价替代，不能据此单独确定或排除某个解码库。

进一步比较相同 rank-0 数据顺序、micro batch 8、一个 spawn worker。各组在第 90 个 batch，即 720 个样本处的结果如下。回收操作仅用于诊断，在子进程内每 64 个样本执行一次。

| 路径 | worker RSS（MiB） | worker USS（MiB） | 趋势 |
|---|---:|---:|---|
| 原 stateful loader | 6022 | 5352 | 持续增长 |
| 普通 PyTorch DataLoader | 4864 | 3938 | 持续增长 |
| stateful + Python GC | 5343 | 4416 | 持续增长 |
| stateful + `malloc_trim(0)` | 1926 | 1003 | 本次短测呈有界波动 |
| stateful + GC + `malloc_trim(0)` | 1912 | 986 | 本次短测呈有界波动 |
| stateful + `MALLOC_ARENA_MAX=2` | 4110 | 3186 | 占用较低，但仍增长 |

`malloc_trim` 单独就能压低占用，而 Python GC 单独不能，说明这段增长的主要机制是数据处理进程的原生堆空闲页保留/碎片。普通 loader 同样增长，因此不能把原因全部归为新增的 stateful 恢复机制。这与可由系统回收的文件页缓存不同：需要进程内分配器释放空闲页，删除磁盘权重不能解决。

已确认的链路是“数据进程原生堆保留导致 RAM 持续增长 → 跨过 Ray 阈值 → 两个 actor 被终止”。尚未将分配行为定位到某个依赖的具体调用；诊断性回收也不等于生产修复已完成。仅设置 `MALLOC_ARENA_MAX=2` 在本次对照中仍持续增长，不能把它当作已经验证的解决方案。原始对照见 [CPU 内存样本](artifacts/loader_memory_diagnostics.csv)。

## checkpoint 与下一次启动条件

本次运行目录没有 `checkpoints/global_step_*`。保存间隔为 2500，而退出发生在 506，因此这 506 步没有可恢复的完整状态。W&B 指标不能重建训练权重；修复后需从基础模型重新开始本次正式实验。

重启前先对真实数据读取做有内存上限的持续测试，覆盖至少原失败运行的每 worker 读取量，并验证内存趋稳。随后重新验证训练、保存和恢复的容量。可以比较减少 worker 数和原生 allocator 参数，但改变 worker 数会改变 stateful loader 的恢复签名，不能宣称与旧配置等价恢复。本轮正式实验用于长时间双卡工程稳定性验证，保存间隔保持 2500，持续记录并定位内存、通信、数据读取和 checkpoint 相关问题。

之前的 20 步训练和 7 步连续保存测试证明了局部吞吐及保存恢复契约，未覆盖一个小时的数据读取过程。它们不足以支持“长期正式训练已准备好”的判断。这次失败将持续数据读取的内存验证补为正式训练前的必要检查。

归档见 [故障审计](artifacts/formal_run_506_failure.json) 和 [主机内存样本](artifacts/formal_run_host_memory.csv)。历史性能结果仍保留其测量意义，当前完整 recipe 的长程稳定性尚未通过。

后续实现与验证单独记录在[空闲堆页回收修复](2026-10-28-worker-heap-reclaim.md)，本页保留故障发生时的配置和判断。
