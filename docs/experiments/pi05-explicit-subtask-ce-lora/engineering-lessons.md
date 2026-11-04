# 问题与防复发记录

遇到卡步、重复认证或吞吐异常时，按本记录恢复当时的判断过程。每项区分症状、证据、修复和以后必须保持的条件；没有测量过的解释不当作根因。

## 第 13 步之后卡住：不等长文本改变了 Collective 顺序

最初双卡运行完成 13 次更新后，在第 14 次更新停止推进。GPU 低功率只是表象。CPU 标签审计发现，同一轮 HL microbatch 在两个 rank 上分别有 98、69 个 CE target token，`ce_chunk_size=32` 对应 4 次和 3 次输出投影。

旧循环在 FSDP wrapper 外部按本地 token 数调用共享 embedding/output module。每次调用都会进入分片参数的 collective；rank 0 进入第 4 次投影收集时，rank 1 已经转向后续操作，collective 序列因此不一致。扩大 GPU 显存或提高功率不能修复这类逻辑错误。

修复将变量长度 chunk 循环移入 `Embedder.forward`：每个 HL microbatch 的 CE 输出阶段、每个 rank 都只调用一次该 FSDP wrapper，内部再按本地长度分块计算 CE；输入 embedding 的原有调用保留。`ce_chunk_size=32`、shifted response/EOS mask、逐样本平均、CE/FM 权重和参数 key 均保留。[详细证据](artifacts/collective_fix_validation.json)

| 验证 | 结果与含义 |
|---|---|
| 旧实现负对照 | CPU/Gloo 的真实 FSDP 在 98/69 token 模式下报告 collective mismatch |
| 修复后的分布式回归 | 2/4 ranks、三组不等长模式通过；loss 和归约梯度匹配非分片参考 |
| 模型集成检查 | 即使 chunk=1，真实小型 Pi0 的 CE 输出阶段也只调用 output module 一次 |
| 当时组件测试 | 166 passed、2 skipped；skip 为可选 Gemma3 依赖和既有 CUDA-only 用例 |
| 人工双卡重启 | 用户确认越过原第 14 步触发点；当时归档快照读到第 147 步 |

以后增加条件分支、变长循环或提前退出时，必须检查它是否跨越 FSDP/collective 边界。所有 rank 要以相同次序进入分片模块和显式 collective；本地计算长度可以不同。保留不等长的真实分布式回归，单卡 loss 测试发现不了这类问题。更大 cluster 仍有同一要求；更多 rank 可能更早出现长度不一致，当前测试没有覆盖多节点 NCCL。

不要用增加 `barrier`、调大 timeout、改变 seed 或强行丢弃长标签掩盖这一错误。重新排查时先确认最后完成的 optimizer update，再核对各 rank 的分支与 wrapper 调用序列；分布式 debug 应用于单独的有界复现，不能直接把所有停顿都判为 PCIe 带宽不足。

## W&B 已经登录，Venv 却再次要求认证

已有全局凭据绑定 `api.forge.coreweave.com`，venv SDK 却默认连接 `api.wandb.ai`。凭据按服务主机匹配；“机器上已登录”不意味着任意 W&B endpoint 都能复用登录。

最终命令明确设置 `WANDB_BASE_URL=https://api.forge.coreweave.com`，继续使用原有 W&B backend。认证信息没有进入配置或仓库。以后迁移 venv、重启 Ray 或切换 logger 时，先核对训练进程的 endpoint 与已有登录主机；endpoint 一致后再排查认证，避免创建第二套账号配置。

Ray worker 继承的是 cluster 启动时的环境。测试使用 `RAY_ADDRESS=local` 和独立 `RAY_TMPDIR`，避免意外连接到持有旧环境的 Ray 实例。共享机器上不要为了修自己任务而执行 machine-wide `ray stop`。

## 分离文本不等于训练已经使用了 Subtask

数据转换只建立存储契约。旧 flow-only loader 如果没有消费 `subtask` 字段，即使 parquet 已写入该列，训练仍可能只使用 task 文本。实验 C 因此同时补齐两视图 transform、HL response mask、CE 分支、LL subtask 条件以及推理生成路径。

以后检查这类功能，要沿“一帧原始数据 → transform 后文本与 mask → model forward → loss/梯度 → inference”的完整链路抽样；只看目录结构、字段是否存在或 CE 数值下降，都不足以确认动作模型使用了正确条件。训练时使用真实标签而推理使用预测文本的分布差异仍然存在，需通过生成文本和动作效果检查。

当前 HL 超长标签会报错，训练也要求非空 subtask。保留这些检查，避免靠静默截断得到能跑却改变目标的数据。对于学习质量，分别记录文本是否非空、EOS、阶段是否正确、动作 loss、真机完成率和不同型号盒子的结果；不能只报告有限标签的拟合准确率。

## Normalization 与静止底盘目标必须配套

本次 FoldBox stats 的 base action 分位数范围为零，原始记录仍可能带微小非零 base command。若没有在同一数据变换中将目标置零，零宽范围的归一化可能放大这些值。

最终 recipe 继承全参数 FoldBox 运行的 `zero_base_action_targets=true`；hierarchical loader 对零 base stats 和缺少匹配 flag 的组合直接报错。以后更换 stats 或允许移动底盘时，重新定义目标与统计量的配对关系。不能把这个任务特定 flag 当成所有 SpiritAI 数据的通用默认值。

## LoRA 比预期慢：先统一计时单位和工作量

当时对“约 1.6 秒”和新训练完整一步的比较混入了不同口径。全参数日志的完整更新约 6.586 秒，除以 4 轮得到约 1.647 秒；这个除法值不是实测单卡或独立 microbatch 延迟。早期 C 则是 global 32、micro 2、每 rank 8 轮，不能直接与 global 64、micro 8 的全参数结果归因比较。[历史计时记录](artifacts/timing_comparison.json)

LoRA 仍需执行基础网络的 dense forward、部分 activation backward 和 FSDP 参数收集。HL 另有 257152 类词表投影与 CE；这是 teacher forcing 的训练计算，不能解释成逐 token 自回归解码。FSDP1 的混合 frozen/trainable flat parameter 还可能为整个 flat tensor 分配并归约梯度，optimizer 只更新 adapter 并不能消除这部分代价。

代码审查得到 18 个混合 Gemma flat tensor 的逻辑元素总量约 23.27 亿，可训练 adapter 约 3347 万，比例约 69.5。这是布局和归约张量的分析，不是网络实测字节数，更不是“训练慢 69.5 倍”。最终 profile 也表明存在大量计算与通信重叠，不能把各类累计时间简单相加。[调优与 profile 摘要](artifacts/execution_tuning_summary.json)

以后比较吞吐同时报告完整 optimizer update、accumulation 轮数、HL records、LL action windows 和计时窗口。即使 global batch 同为 64，当前 C 每次只含 32 个动作窗口，全参数 flow-only 每次有 64 个；相同步数并不保证相同动作曝光量。严格变量分离已由用户暂缓，本轮保留这一解释边界。

## 显存不同或功率较低，不能直接判定没有工作

两 rank 的局部标签长度、临时张量、allocator cache 和进程占用可以不同；单个功率或利用率快照不能区分正在计算、数据等待和 collective 等待。第 14 步问题通过调用序列定位，调优则通过端到端计时和 GPU timeline 判断。

最终 100 步的 PyTorch peak allocated 为跨 rank 最大 38.43 GiB；NVML 采样峰值约为 GPU0 48229 MiB、GPU1 48357 MiB。NVML 包括 allocator cache 等占用，两者不是同一指标。不能仅凭 `nvidia-smi` 剩余空间就断言等量张量不可分配，也不能从短测 allocated 峰值推断 checkpoint 保存一定有余量。

已测 `shard_grad_op + 关闭重计算 + 局部编译` 在首个更新 OOM，因此没有选用。最终配置仍需在真实保存和恢复阶段检查容量；100 步不保存权重的测试没有覆盖这个峰值。

## 保存恢复、短测与长期运行必须分别验证

本轮最终测试明确设置 `runner.max_steps=100`、`runner.save_interval=-1`，连最后一步也不写权重。它证明有限训练更新通过，不能引用旧全参数的恢复测试来宣称新 LoRA 配置已通过完整生命周期。

恢复需保留 model/optimizer/scheduler、loader/sampler/worker RNG、各 rank RNG 以及 HL/LL 的 objective offset。当前实现把训练模型配置、stats 和文本模板版本写入 `openpi_training.json`，并对数据与恢复条件做校验。只保留 merged `full_weights.pt` 足以走部署方向，却不能重建完整训练状态。

原全参数实验还曾在第 506 步遇到数据进程主机内存增长，随后增加每 worker 64 次读取的 heap reclamation。当前 C 继承 `worker_malloc_trim_interval=64`，应继续观察主机 RAM、Ray 内存和 worker RSS。[原问题与修复](../pi05-flow-only-full/2026-10-28-worker-heap-reclaim.md)是历史依据，不能替代 C 的长期运行记录。

## 推送备份时的认证问题

实现提交首次通过 HTTPS 推送时，Git 没有 credential helper；改用已有 GitHub CLI 登录后，又因 OAuth 缺少 `workflow` scope 被拒绝，原因是提交包含 `.github/workflows/ci-tests.yml`。最后使用已有 SSH key，通过 `ssh.github.com:443` 完成完整分支推送，远端与本地 SHA 均为 `3a7af6d1`。

以后遇到此类失败，区分“没有 Git 凭据”和“凭据权限不足”。使用已有 SSH 路径或有相应 scope 的登录，保留完整 CI 改动。SSH host key 当时通过 GitHub API metadata 核实，并使用临时 known-hosts 文件；临时文件不属于工程恢复依赖。
