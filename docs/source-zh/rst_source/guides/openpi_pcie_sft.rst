调优 PCIe 多卡 OpenPI SFT
===========================

当多卡 OpenPI 全参数 SFT 受到 PCIe 通信和内存容量限制时，按本指南测量和调优。从 :doc:`OpenPI SFT 配方 <../examples/embodied/sft_openpi>` 出发，先核对实际传输路径，再增大每卡 micro batch，最后验证 checkpoint 恢复，确定长训练预算。

以下实测来自 2026-10-27 的 SpiritAI FoldBox 实验：两张 RTX PRO 5000 Blackwell，每卡报告 48,935 MiB，主机约 125 GiB RAM，模型共 3,353,433,872 个参数。数据为三路相机和 25D Cartesian 布局，动作 pad 至网络 32D，horizon 为 50，prompt/state token 上限为 256。这些数值描述一台机器的容量和吞吐；收敛与真机任务成功率需要另行验证。

先建立可比较的基线
--------------------

先固定 global batch，再讨论每步耗时是否因某项优化而下降。启用梯度累积时，一次 optimizer 更新包含多轮前向和反向：

.. math::

   G = N_{GPU} \times M \times A

其中 ``G`` 为 global batch，``M`` 为每卡 micro batch，``A`` 为累积次数。两卡 global batch 64、micro batch 1 需要每卡 32 轮；micro batch 8 则为 4 轮。只增大 global batch 会增加累积次数，不会增大每次 GPU 计算的批量。

对照性能时，保持模型、观测、horizon、归一化、seed、优化器和精度策略一致。本次采用 FP32 参数存储与更新、BF16 主干计算、FP32 归约、已有 OpenPI FP32 模块策略、梯度检查点，以及 FSDP1 ``full_shard``。局部算子编译关闭，以便聚焦 PCIe。框架支持 FSDP2，但当前版本的 OpenPI 分片路径校验要求 FSDP1，不能将切换后端当作即插即用的调优步骤。有限测试使用恒定学习率 ``1e-5`` 和 warmup 0 以观察更新；正式计划为 warmup 750，吞吐检查并不验证该优化 schedule。

通用 SpiritAI YAML 保留 global batch 32、micro batch 1 的保守默认值。记录包含 CLI 覆盖的最终解析配置；仅选择这个 YAML 名称，并不会得到调优后的 64/8 配置。

确认实际通信路径
------------------

GPU 支持 peer access，不代表 NCCL 一定选择 P2P。先读取拓扑和读写能力，再检查有限训练运行实际使用的通道：

.. code-block:: bash

   nvidia-smi topo -m
   nvidia-smi topo -p2p r
   nvidia-smi topo -p2p w
   nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv

   mkdir -p logs
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,ENV
   export NCCL_DEBUG_FILE="$PWD/logs/nccl.%h.%p.log"

这些命令读取硬件，并为下一次启动生成分进程 NCCL 诊断文件。本机两卡链路均为 Gen5 x16，拓扑为同 NUMA 内的 ``NODE``，读写 P2P 均可用，但初始 GPU 通道使用 ``SHM/direct``。仅凭初始化消息 ``Using network Socket``，不能判断本机 GPU 数据实际走哪条路径。

针对该拓扑，允许 PHB peer access 后，通道变为 ``P2P/IPC``：

.. code-block:: bash

   export NCCL_P2P_LEVEL=PHB

PHB 允许同 NUMA 节点内经过主桥的 peer 通信，并不会建立 NVLink 连接。将它保留为经过测量的本机启动设置；换机器时重新检查正确性和吞吐。NCCL 默认会根据架构和环境选择距离阈值，详见 `NCCL 传输配置 <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level>`_。

解释带宽时，要区分字节单位和指标口径。26.7 GB/s 约为 214 Gb/s；PCIe Gen5 x16 理论单向约 63.0 GB/s，尚未扣除事务层开销。本机实测单向 Tensor 拷贝为 41.3–41.7 GB/s。双 rank all-gather 和 reduce-scatter 使用 ``busbw = algbw × (n-1)/n``，所以 bus bandwidth 是 algorithm bandwidth 的一半；它们与单向拷贝并非同一负载。参考 `PCI-SIG 带宽表 <https://pcisig.com/sites/default/files/files/PCI-SIG-Seamless_Transition_to_PCIe_5.0_in_System_Implementations_FINAL.pdf>`_ 和 `NCCL Tests 计量定义 <https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md>`_。

减少重复参数传输
------------------

确认 P2P 后，保持 global batch 不变，逐步增大 micro batch。它会减少重复参数 all-gather 和调度开销，并改变矩阵计算规模；应测量组合效果，而不能把所有增益都归为 PCIe 带宽。

.. figure:: ../../../experiments/pi05-flow-only-full/artifacts/pcie_sft_tuning.svg
   :width: 100%
   :alt: OpenPI SFT 受控步耗时及采样显存余量对照。

   每组运行 8 次更新，排除前 3 次。误差范围为后 5 次的最小到最大值，不是置信区间。NVML 占用包含桌面进程，也可能漏掉更短的峰值。

.. list-table:: 固定 global batch 64
   :header-rows: 1

   * - 传输
     - 每卡 micro batch
     - 每卡累积次数
     - 每次更新的平均训练秒数
   * - SHM
     - 1
     - 32
     - 28.056
   * - P2P
     - 1
     - 32
     - 24.448
   * - P2P
     - 2
     - 16
     - 13.514
   * - P2P
     - 4
     - 8
     - 8.605
   * - P2P
     - 8
     - 4
     - 6.526
   * - P2P
     - 32
     - 1
     - 5.846

同 micro batch 1，仅改变传输路径，训练耗时下降约 12.9%。选定的 micro batch 8 另做了 20 次更新确认，后 17 次平均为 6.695 秒，相对初始基线吞吐约提高 4.19 倍。原生 ``time/training`` 包含数据等待、前后向和 optimizer 更新，不含 checkpoint 写入和模型初始化。

按完整资源生命周期选择配置。micro batch 32 在短测中达到 48,151 MiB，仅余约 784 MiB 采样容量；最终选择并验证保存与重启的是 micro batch 8。该配置在恢复时达到 48,130 MiB，连续保存测试达到 48,276 MiB，因此稳态训练样本不能代表保存或重启的峰值。

本机其他已有开关没有改善最终取舍：backward prefetch 仍接近 28 秒；``shard_grad_op`` 的 micro batch 1 为 15.180 秒、峰值 48,327 MiB。固定 NCCL 为 16 CTA 可把 256 MiB all-gather 短测提高到约 29.7 GB/s，但实际 micro batch 8 训练仅从 6.526 变为 6.501 秒，峰值却多了 1,163 MiB，因此保留默认 CTA 自动选择。

相同 micro batch 下，传输、预取和分片对照的 loss 与梯度范数记录一致。改变 micro batch 会改变随机噪声、时间和增广的抽样分组，因此不同 micro batch 的 loss 序列不要求逐步相同。短测中的有限 loss 也不能证明收敛。若需区分 kernel、通信和数据等待的占比，使用 :doc:`Profiling <profile>`；本轮采用传输日志和受控运行，没有采集 GPU timeline。

验证恢复与内存回收
--------------------

仅验证权重能推理，还不能证明训练可恢复。保存完整训练状态，退出进程后重新恢复，并核对 optimizer 进度、数据位置、RNG 和后续权重更新。checkpoint 布局见 :doc:`恢复训练 <resume>`。

本轮从第 20 步恢复到 21 步，665 个活跃 optimizer state 推进到 21，两个结构上未参与当前 loss 的末层 prefix norm state 保持 0。每 rank 已消费的 micro batch 从 80 推进到 84，动作头、视觉与语言骨干的代表性权重继续变化。当前未做不中断运行与恢复运行的逐参数等价对照。使用可选的 stateful loader 恢复签名时，保持 micro batch、worker 数、seed、rank 布局、数据和 stats 一致。

主机 RAM 与 GPU 显存需要分别检查。RLinf 将分片训练状态暂存到 CPU，并在 rank 0 导出完整权重。单个 checkpoint 约 50 GiB，其中完整 FP32 权重约 12.5 GiB；磁盘尺寸并不等于同等额外 RAM 需求，主机 swap 也不能解决 GPU OOM。

.. figure:: ../../../experiments/pi05-flow-only-full/artifacts/checkpoint_host_memory.svg
   :width: 100%
   :alt: 四次 checkpoint 保存期间的主机可用 RAM 和匿名内存。

   本组测试使用原有 2 GiB swap。系统可用 RAM 和进程 cgroup 匿名内存统计范围不同，不能相加。

7 次更新的测试设置每 2 步保存，最终步也保存，因此实际文件位于第 2、4、6、7 步。578 个约 0.5 秒间隔的样本中，主机可用 RAM 始终高于 52.53 GiB。匿名内存峰值约为 56.30、57.19、58.05、58.42 GiB，保存后回落，进程退出后系统可用 RAM 回到约 111 GiB。约 2.1 GiB 的增长应保留在记录中；短测不能证明多天运行零增长。

删除审计后的 4 份临时 checkpoint 回收约 199.92 GiB 磁盘，但系统可用 RAM 仍约为 111.8–111.9 GiB，缓存则从约 65 降到 15 GiB。文件缓存本来就可回收。将它与活进程内存、命名 tmpfs 对象和保留的 checkpoint 文件区分，使用 ``MemAvailable`` 以及对应进程/cgroup 计数，不要把全部缓存视为泄漏。统计范围参考 `Linux 内存计数器 <https://docs.kernel.org/filesystems/proc.html#meminfo>`_ 与 `tmpfs 说明 <https://docs.kernel.org/filesystems/tmpfs.html>`_。

测试之后，系统保留原 2 GiB 并新增 30 GiB，swap 总量约为 32 GiB。这项后续运维变更应与通过测试时的条件分开记录。SFT runner 会保留 checkpoint 目录；诊断权重写入失败时，还需检查磁盘保留量。

保留预算与运行证据
--------------------

改变 global batch 时，同时换算样本预算。旧 LoRA 配置为 ``32 × 90000 = 2880000`` 个窗口；global batch 64 对应 45000 次更新，128 对应 22500 次。warmup 和保存间隔也应按相同单位比较。这份继承预算并不是已经验证的全参数收敛要求。

本机手动计划为 scheduler 总计 45000 步、warmup 750 步、每 2500 步保存。首阶段可以在 5000 步停止，并保留 45000 步 schedule 供恢复使用。按确认短测估算，首阶段纯计算约 9.3 小时，完整曝光预算约 3.5 天，另计写入、中断及长程波动。18 份每份约 50 GiB 的 checkpoint 需要约 900 GiB 磁盘。

将最终 Hydra 配置、完整代码版本、归一化/数据指纹、拓扑与 NCCL 路径日志、计时样本及预热规则、checkpoint 审计放在一起。本次 HEAD 还有未提交改动，所以仅记录 commit hash 不足以重建现场。将小型结果归档到被忽略的日志目录之外。W&B 凭据继续使用已有登录；若全局 CLI 和 venv SDK 默认服务不同，将 ``WANDB_BASE_URL`` 与已登录主机名匹配。

用 tmux 应对终端断连，用 checkpoint 应对进程或机器终止。正式训练启动和真机动作仍由人工执行。数据与模型字段见 SFT 指南，实验归档 ``docs/experiments/pi05-flow-only-full/README.md`` 保留了本机完整启动及恢复命令。

下载实测数据
--------------

本机日志清理后，使用已进入仓库的 CSV 重建表格。原始完整 trace 和模型大文件是另外保留的本地产物。本轮有限测试由 AI assistant 协助执行和整理。

* :download:`训练对照 <../../../experiments/pi05-flow-only-full/artifacts/training_benchmarks.csv>`
* :download:`通信短测 <../../../experiments/pi05-flow-only-full/artifacts/communication_benchmarks.csv>`
* :download:`主机内存样本 <../../../experiments/pi05-flow-only-full/artifacts/host_memory_samples.csv>`
* :download:`checkpoint 内存审计 <../../../experiments/pi05-flow-only-full/artifacts/checkpoint_memory.csv>`
* :download:`验证摘要 <../../../experiments/pi05-flow-only-full/artifacts/validation_summary.json>`
