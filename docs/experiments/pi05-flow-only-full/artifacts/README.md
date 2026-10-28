# 归档数据与口径

用这些小型文件重建结果表、核对实际配置，并在清理本机 `logs/` 后保留本次工程判断的依据。数据从原始 W&B history、NCCL 短测 JSON 和 checkpoint 审计提取；这里不打包数据集、模型权重、账号凭据或完整原始日志。

| 文件 | 内容 |
|---|---|
| `experiment_manifest.json` | 测试条件、HEAD 加未提交改动的说明、21 个关键源文件的 SHA-256、数据 metadata/stats 指纹、归档时的 swap 状态 |
| `formal_resolved.yaml` | 手动正式入口 `--cfg job --resolve` 的解析结果；global 64 / micro 8，首阶段 5000、scheduler 45000 |
| `formal_resolved_heap_reclaim.yaml` | 回收修复后手动命令的解析结果；每 worker 64 个样本回收，保持每 2500 步保存 |
| `training_benchmarks.csv` | 9 组 8 步对照、20 步确认、1 次恢复更新及 7 步内存复核，共 12 行 |
| `training_steps.csv` | 上述运行共 100 次更新的原始计时、loss、grad norm、学习率与时间戳 |
| `preflight_summary.json` | 较早 micro 1 的 global batch 32/64/128 预检结果及估算口径 |
| `communication_benchmarks.csv` | 4 种 NCCL 设置 × 5 种消息尺寸 × 2 种 collective，共 40 行 |
| `p2p_copy_benchmarks.csv` | 两个方向分别进行的单向 Tensor 拷贝，不是同时双向通信 |
| `checkpoint_memory.csv` | 第 2/4/6/7 步保存窗口的 RAM、匿名内存、文件完整性及数据位置审计 |
| `host_memory_samples.csv` | 578 个约 0.5 秒间隔的主机与 cgroup 内存样本 |
| `checkpoint_windows.csv` | 从原生 runner 时间估算的保存窗口；用于图中的阴影标注 |
| `validation_summary.json` | optimizer、权重、loader、RNG、离线推理和临时文件清理证据摘要 |
| `formal_run_506_failure.json` / `formal_run_host_memory.csv` | 后续正式运行第 506 步触发 Ray 内存保护的审计、CPU 隔离结果与 225 个主机内存样本 |
| `loader_memory_diagnostics.csv` | 同一真实数据顺序下 6 组 CPU worker 对照，共 54 个采样点；普通/stateful loader、GC、allocator 回收及 arena 设置 |
| `worker_heap_reclaim_validation.json` | 回收修复后的源码指纹、单测、真实数据 8 worker 验证和各进程峰值 |
| `worker_heap_reclaim_soak.csv` / `worker_heap_reclaim_soak.svg` | 每 worker 消费 5000 个样本的 CPU 数据压力验证，253 个采样点及图表 |
| `pcie_sft_tuning.svg` / `checkpoint_host_memory.svg` | 从归档 CSV 生成的图表 |
| `increase_swap_to_32g.sh` | 从既有约 2 GiB swap 新增 30 GiB、备份 fstab 并持久化的人工运维脚本 |

训练耗时取原生 `time/training`，排除 checkpoint 保存。8 步对照丢弃前 3 步，余下 5 步的 mean/min/max 均保留；min/max 不是置信区间。20 步确认取后 17 步。恢复更新只有一步、包含冷启动，不参与稳态估算。GPU 显存是 NVML 总占用的间隔采样，包含桌面进程，不能当作 PyTorch 分配量或连续时间的绝对峰值。

带宽的 `GBps` 为十进制 GB/s，消息尺寸 `full_bytes` 为字节。两 rank collective 的 `bus_GBps_two_ranks` 按 `algorithm_GBps × 0.5` 计算。完整消息量包含本地份额，所以不能把 algbw 当成单向物理链路吞吐。计时包含 Python collective 调用与设备同步；这些是受控短测，不是标准 `nccl-tests` 二进制的输出。

主机 `MemAvailable`、全系统匿名页和进程 cgroup 匿名内存有不同统计范围，不能相加成内存分解。保存窗口依赖 runner 的计时与 W&B 时间戳对齐，是近似边界。四份连续保存测试权重已在审计后删除，文件清单和清理摘要保留在本机日志；JSON 摘要记录这一事实。

`logs/spiritai_preflight_20261027_131340/` 和 `logs/spiritai_tuning_20261027/` 是原始本机证据根目录，受 Git 忽略。SHA-256 指纹用于核对代码和资产内容，不替代完整版本快照；原模型大文件没有在这里计算或归档内容哈希。所有测试发生在约 2 GiB swap 条件下，清单中的约 32 GiB 是归档时已生效的后续系统状态。

清单另外保留归档时的包版本和模块实际路径。本机同时安装 `nvidia-nccl-cu12 2.32.3` 与 `nvidia-nccl-cu13 2.28.9`；短测的 Torch API 报告后一个版本，NCCL 初始化日志报告前一个版本。两种观察都予以保留，不凭其中一个版本号推断所有运行时链接细节。

提交前统一了 CSV 换行、SVG 行尾空白和 YAML 末尾空行；CSV 内容与 YAML 解析值逐项核对一致。清单的源文件指纹对应最终归档字节，另存两个仅格式变化的 YAML 在测试时的原指纹，保留测量版本的依据。

## 归档校验

整理时重新计算并核对了 21 个源文件指纹、12 组运行的计时均值及样本数、40 行 collective 结果和 578 个内存样本。中英文生成页面各 245 个本地链接及图片目标均存在。正式启动命令以 `--cfg job --resolve` 解析，19 个关键配置值与归档 YAML 一致；该检查不启动训练。swap 脚本通过 `bash -n`。

`python3 .agents/skills/docs-check/build_docs.py --keep-output` 的中英文检查均为零告警；该 docs-only 检查器按默认规则过滤未安装模型依赖导致的 autodoc 导入告警。受影响页面的 RST markup 检查、新指南的符号检查和 `git diff --check` 均通过；符号检查显式放行外部系统字段 `MemAvailable` 与 W&B 环境变量 `WANDB_BASE_URL`。
