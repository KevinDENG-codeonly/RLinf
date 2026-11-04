# 工程证据与历史口径

用此目录中的小型 JSON 核对实验 C 的关键事实，即使本机 `logs/` 或 `/tmp` 被清理，也能保留配置、问题触发条件与性能结论。数据、视频、checkpoint 和完整 profiler trace 需要在外部单独保留。

## 文件说明

先按阶段选择证据。历史记录是当时的快照；里面的保存间隔、batch 或已完成步数不能直接套到最终 recipe。

| 文件 | 来源 / 内容 | 边界 |
|---|---|---|
| [data_conversion_report.json](data_conversion_report.json) | 2026-10-30 临时转换目录中的原始报告；字段变更、label 分布、保留项 | 转换时报告，源 metadata 的既有差异被保留 |
| [data_reader_smoke_test.json](data_reader_smoke_test.json) | 当时官方 LeRobot reader 的抽样与图像解码 | temp staging 路径为历史位置；最终目录见报告 destination |
| [collective_fix_validation.json](collective_fix_validation.json) | 第 14 步不等长 CE/FSDP 故障、CPU 负对照、2/4 rank 修复回归 | 当时为 global 32/micro 2、warmup 1500、save 2500；GPU 快照只到 147 步 |
| [timing_comparison.json](timing_comparison.json) | Full FT 与早期 C 的 W&B updates 11–80 | 不同 batch/目标/可训练范围的描述性比较，不是 LoRA 因果消融 |
| [aligned_100_step_validation.json](aligned_100_step_validation.json) | global 64/micro 8、45k scheduler、warmup 750 的基线 100 步 | 当时计划 save 2500；测试实际 save=-1，历史值保留 |
| [selected_execution_100_step_validation.json](selected_execution_100_step_validation.json) | 最终三个执行开关、100 条 history、51–100 同窗口比较 | 正式 save 5000；测试无保存，全部在 warmup 内 |
| [execution_tuning_summary.json](execution_tuning_summary.json) | 2026-11-04 从已有本机摘要补档的执行候选与 GPU timeline 摘要 | 单次运行；第 6 步 profile 被排除计时；不等同于完整 trace |
| [engineering_backup_manifest.json](engineering_backup_manifest.json) | 实现 commit、关键文件/metadata/stats 指纹、环境版本、补档核对范围 | 指纹只覆盖明确列出的文件，未打包完整数据或基础权重 |

## 如何引用这些结果

最终性能引用 selected 100 步与 aligned 100 步的同窗口结果：6.5097 → 4.8313 秒/完整更新。用 `time/training`，并同时报告 32 HL + 32 LL 的目标曝光量；不要把它写成单卡或单个 microbatch 延迟。

`peak_allocated_gib` 与 NVML `sampled_peak_gpu_memory_mib` 使用不同口径。profile 的 kernel 并集、通信并集和重叠时间也不能重复相加。compiled BF16 引起的小数差异保留在 selected 的 `max_abs_metric_difference_vs_baseline` 中。

本次补档直接复制两份历史数据报告，新增调优摘要与清单，保留原四份验证 JSON 不变。文档没有把历史的 2500 保存计划改写成 5000，也没有将 147 步快照改写成之后的完成状态。正式运行后应新增结果记录，并标明证据时间与配置。
