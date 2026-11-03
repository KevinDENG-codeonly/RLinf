OpenPI 监督微调
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ 视觉-语言-动作模型。

在 RLinf 里对 π₀ / π₀.₅ 做 **全量 SFT** 或 **LoRA**。策略统一为 ``model_type: openpi``：这是与 JAX 参考对齐的 PyTorch 实现（官方 openpi 仓库自带的 PyTorch 代码并未对齐）。SFT 通常是强化学习前的冷启动：先模仿示范，再在较好先验上做 RL。

配方一览
----------------------------------------

实验配置通过 Hydra ``defaults`` 引入模型模板，再覆盖路径和任务字段。启动前把 ``/path/to/...`` 换成本地路径。

.. list-table::
   :header-rows: 1
   :widths: 18 38 22 22

   * - 任务
     - 实验配置（``examples/sft/config/``）
     - 模型模板
     - 启动名
   * - LIBERO π₀
     - ``libero_sft_openpi.yaml``
     - ``model/pi0``
     - ``libero_sft_openpi``
   * - 真机 Franka bin-relocation
     - ``realworld_bin_relocation_sft_openpi.yaml``
     - ``model/pi0``
     - ``realworld_bin_relocation_sft_openpi``
   * - 自定义 LeRobot
     - ``custom_sft_openpi.yaml``
     - ``model/pi0``
     - ``custom_sft_openpi``
   * - BEHAVIOR-1K π₀.₅
     - ``behavior_sft_openpi_pi05.yaml``
     - ``model/pi0_5``
     - ``behavior_sft_openpi_pi05``
   * - RoboTwin adjust_bottle π₀
     - ``robotwin_adjust_bottle_sft_openpi.yaml``
     - ``model/pi0``
     - ``robotwin_adjust_bottle_sft_openpi``
   * - SpiritAI FoldBox 显式 subtask
     - ``spiritai_sft_openpi_pi05_subtask_lora.yaml``
     - ``model/pi0_5``
     - ``spiritai_sft_openpi_pi05_subtask_lora``

双 Franka 真机的 SFT 与部署见 :doc:`dual_franka_openpi_pytorch`。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: 方法
      :text-align: center

      Full SFT · LoRA

   .. grid-item-card:: 数据
      :text-align: center

      LeRobot · BEHAVIOR

   .. grid-item-card:: 硬件
      :text-align: center

      1+ 节点 · GPU

| **你将完成：** 安装依赖 → 准备数据与归一化统计 → ``bash examples/sft/run_vla_sft.sh <启动名>`` → 看训练损失。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 对应格式的数据集。

通用约定
----------------------------------------

训练时 ``openpi.task`` 必须为 ``sft``；评测 YAML 用 ``eval``，不要把训练的 ``sft`` 抄过去。

``num_action_chunks`` 和环境实际执行的 chunk 是一回事，**不是**\ 网络 horizon：

- ``num_action_chunks`` / ``openpi.action_chunk``：环境一次执行多长。
- 网络 ``action_horizon``：优先用 YAML 的 ``openpi.action_horizon``；否则用 ``openpi.config_name`` 对应官方 ``TrainConfig.model.action_horizon``。默认模板只把 ``num_action_chunks`` 插值到 ``action_chunk``，**不会**\ 把它写成网络 horizon。

官方 LeRobot SFT 的数据窗口也来自 ``TrainConfig.model.action_horizon``，不读 ``num_action_chunks``。常见官方 horizon：BEHAVIOR ``pi05_behavior`` **32**\ ，RoboTwin ``pi0_aloha_robotwin`` / LIBERO ``pi0_libero`` **50**\ ，真机 ``pi0_realworld`` **10**。只有 checkpoint 的 horizon 和该 ``TrainConfig`` 不一致时，才覆写 ``openpi.action_horizon``。

BEHAVIOR / dual-Franka 的流式 SFT 加载器目前仍用 ``num_action_chunks`` 当数据窗口，这两份配方里才需要把它设成和官方 horizon 一样。

设置 ``actor.model.precision: fp32``，让原始参数和优化器更新使用 FP32。
设置 ``actor.model.openpi.embed_dtype: bf16``，让主干激活和残差使用 BF16。
``embed_dtype`` 未设置或为 null 时沿用 ``model.precision``；
``precision: null`` 对应 BF16 激活。
单独配置 FSDP：前向和反向计算使用 BF16 参数，梯度归约和 buffer 使用 FP32。
共享模型模板使用 ``precision: null``，采用 OpenPI 的选择性 BF16/FP32 参数 dtype。
BEHAVIOR、RoboTwin、dual-Franka 和 RLT Stage 1 SFT 配方设置 ``precision: fp32``。

选择 ``hybrid_engines/openpi_fsdp``，让 projection、RMSNorm 权重和视觉 embedding
使用 FP32。该配方保留 FP32 的 action/time 输入及自适应 norm 条件，主干使用 BF16 计算。
RLT Stage 1 配方还让 ``rlt_module`` 使用 FP32。

OpenPI 支持 FSDP1 的 ``no_shard``、``shard_grad_op`` 和 ``full_shard``，
均使用 ``use_orig_params: true``。``no_shard`` 复制全部训练状态；
``shard_grad_op`` 对梯度和优化器状态分片；``full_shard`` 还在模块执行之间对参数分片。
选择分片模式可降低每个 rank 的显存占用。设置 ``gradient_checkpointing: true`` 和
``gradient_checkpointing_use_reentrant: false`` 可进一步降低激活显存。
BEHAVIOR / RoboTwin 配方使用 ``openpi_cosine`` 学习率调度。

.. code:: yaml

   defaults:
     - model/pi0_5@actor.model
     - hybrid_engines/openpi_fsdp@actor.fsdp_config
     - _self_

   actor:
     model:
       precision: fp32
       openpi:
         task: sft
         embed_dtype: bf16
     fsdp_config:
       sharding_strategy: no_shard
       use_orig_params: true
       mixed_precision:
         param_dtype: bf16
         reduce_dtype: fp32
         buffer_dtype: fp32

Tokenizer 由 OpenPI 的 ``ModelTransformFactory`` 按基础模型加载，YAML 里不用单独写 SentencePiece 路径。

数据集
----------------------------------------

用 ``actor.model.openpi.config_name`` 选择数据格式：

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - ``config_name``
     - 数据集 / 环境
   * - ``pi0_libero`` · ``pi05_libero``
     - LIBERO
   * - ``pi0_realworld``
     - 真机 Franka
   * - ``pi05_yam_joint``
     - 真机双臂 YAM
   * - ``pi0_aloha_robotwin``
     - RoboTwin（ALOHA）
   * - ``pi05_behavior``
     - BEHAVIOR-1K（流式加载）
   * - ``pi0_maniskill`` · ``pi05_maniskill``
     - ManiSkill
   * - ``pi05_metaworld``
     - MetaWorld
   * - ``pi05_calvin``
     - CALVIN
   * - ``pi0_custom``
     - 自定义 LeRobot

自定义 LeRobot 数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 在 ``examples/sft/config/custom_sft_openpi.yaml`` 里指定 ``config_name``：

   .. code:: yaml

      actor:
        model:
          openpi:
            task: sft
            config_name: "pi0_custom"

2. 在 :file:`rlinf/models/embodiment/openpi/dataconfig/__init__.py` 注册 ``pi0_custom``\ （仓库里已有一份模板）：

   .. code:: python

      TrainConfig(
          name="pi0_custom",
          model=pi0_config.Pi0Config(),
          data=CustomDataConfig(
              repo_id="physical-intelligence/custom_dataset",
              base_config=DataConfig(prompt_from_task=True),
              assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
              extra_delta_transform=False,
              action_train_with_rotation_6d=False,
          ),
          pytorch_weight_path="checkpoints/torch/pi0_base",
      ),

3. 按自己的 key / 动作空间改 ``CustomDataConfig``，定义在 ``rlinf/models/embodiment/openpi/dataconfig/franka_dataconfig.py``。

归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

新采集的 LeRobot 数据要在 SFT 前算 ``state`` / ``actions`` 的 ``norm_stats``，真机数据尤其需要：

.. code:: bash

   # 本地数据集目录（含 meta/info.json）：
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id /path/to/realworld_franka_bin_relocation

   # 或 Hugging Face repo id（默认缓存在 ~/.cache/huggingface/lerobot）：
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

- ``--repo-id`` 可以是本地路径或 LeRobot HF repo id；可用 ``HF_LEROBOT_HOME`` 改缓存父目录。
- ``--config-name`` 必须和训练用的 dataconfig 一致。
- ``calculate_norm_stats.py`` 默认写到 ``./assets/<config_name>/<repo_id>/norm_stats.json``\ （``TrainConfig.assets_dirs / repo_id``）。训练和评测\ **不会**\ 自动从 SFT 的 ``full_weights.pt`` 目录找这份文件：要用 ``openpi_data.norm_stats_path`` 显式指过去。省略该字段时，OpenPI 按 TrainConfig 默认读取 ``{assets_dir}/{asset_id}/norm_stats.json``\ （会在 ``model_path`` 覆盖之后），并打一条 warning。RoboTwin 配方把统计钉在基础权重旁的 ``physical-intelligence/robotwin/<task>/norm_stats.json``。

若标准差过小或 q99–q01 过窄，适当放大通常更稳，尤其是 SFT 之后还要做在线训练时。

BEHAVIOR 不跑上面的脚本，而是用 ``actor.model.openpi_data.norm_stats_path`` 指向已有的 ``{assets}/{asset_id}/norm_stats.json``。

安装
----------------------------------------

.. include:: _setup_common.rst

**方式一：Docker** —— 镜像 ``agentic-rlinf0.4-maniskill_libero``：

.. code:: bash

    docker run -it --rm --gpus all \
        --shm-size 20g \
        --network host \
        --name rlinf \
        -v .:/workspace/RLinf \
        rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
        # 国内镜像：infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

    source switch_env openpi

**方式二：自建环境**：

.. code:: bash

    # 国内可加 --use-mirror
    bash requirements/install.sh embodied --model openpi --env maniskill_libero
    source .venv/bin/activate

``--model openpi`` 装的是官方 OpenPI **运行环境**\ （transforms、数据管线）；训练用的网络仍是 ``openpi``。

LIBERO 与真机 Franka
----------------------------------------

先改实验 YAML 里的数据路径和 ``actor.model.model_path``。常用覆写：

.. code:: yaml

   data:
     train_data_paths: /path/to/libero-data

   cluster:
     num_nodes: 1
     component_placement:
       actor,env,rollout: 0-0

   actor:
     model:
       model_path: /path/to/pi0-model
       num_action_chunks: 4          # 环境执行长度，不是网络 horizon
       is_lora: False                # LoRA 时改为 True，并设 lora_rank
       openpi:
         task: sft
         embed_dtype: bf16
         config_name: "pi0_libero"   # 真机改为 pi0_realworld

启动：

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openpi
   bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

日志和 checkpoint 写在 ``runner.logger.log_path`` 下，每 ``runner.save_interval`` 步保存到 ``.../checkpoints/global_step_<N>/``。

Pi0.5 基础 checkpoint 下载与转换
----------------------------------------

π₀.₅ 的 JAX 基座位于 ``gs://openpi-assets/checkpoints/pi05_base``。下载后的目录包含
``params/``，需要先转换为 RLinf OpenPI 格式，才能填入 ``actor.model.model_path``。
YAML 中的 ``/path/to/...`` 是占位路径，请替换为服务器上的实际目录。

在 RLinf 仓库根目录、安装了 JAX / Orbax 的 OpenPI 环境中执行以下命令；下载需要 Google Cloud CLI：

.. code:: bash

   mkdir -p /path/to/checkpoints/jax
   gcloud storage cp --recursive \
     gs://openpi-assets/checkpoints/pi05_base /path/to/checkpoints/jax/

先准备当前任务的 ``norm_stats.json``\ （见上文归一化统计），再转换。以下以 BEHAVIOR 的
32-step horizon 为例；``--input-model`` 应指向包含 ``params/`` 的父目录：

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/checkpoints/jax/pi05_base \
     --input-norm-stats /path/to/task/norm_stats.json \
     --output-model /path/to/pi05_base_openpi \
     --output-norm-stats /path/to/pi05_base_openpi/task/norm_stats.json \
     --action-dim 32 \
     --action-horizon 32 \
     --max-token-len 200

转换器默认处理 π₀.₅，不要添加 ``--no-pi05``。其他 π₀.₅ SFT 配方使用相同流程，
转换参数中的 ``--action-horizon`` 按已有配置填写：ManiSkill RLT 为 10，Franka RLT 和
dual-Franka 为 20。无需为转换修改 YAML 中的动作维度、图像数或训练参数。

确认输出目录包含 ``model.safetensors`` 和 ``config.json``，然后把
``actor.model.model_path`` 的占位值替换为转换输出目录，并将
``actor.model.openpi_data.norm_stats_path`` 指向同一任务的统计文件，例如：

.. code:: yaml

   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         norm_stats_path: /path/to/pi05_base_openpi/task/norm_stats.json

已经是 RLinf OpenPI 格式的 checkpoint 无需重复转换。官方 OpenPI PyTorch 格式
（权重键为 ``paligemma_with_expert.*``）即使有 ``model.safetensors``，仍需使用
``openpi_pytorch_to_openpi`` 转换；命令见 ``rlinf/utils/ckpt_convertor/openpi/README.md``。

Pi0.5 + BEHAVIOR-1K
----------------------------------------

路径写在实验配置里，模型形状来自 ``model/pi0_5``：

.. code:: yaml

   defaults:
     - model/pi0_5@actor.model
     - hybrid_engines/fsdp@actor.fsdp_config

流式加载器只读 ``data:``，没有隐藏默认值。至少改这些路径：

.. code:: yaml

   data:
     train_data_paths: /path/to/2025-challenge-demos
     behavior_dataset_root: /path/to/2025-challenge-demos
     repo_id: "behavior-1k/2025-challenge-demos"
     modalities: ["rgb"]
     num_workers: 8
     tasks: ["turning_on_radio"]
     use_skill: false
     task_subtasks:
       turning_on_radio:
         - "move to radio"
         - "pick up radio from coffee table"
         - "press radio"
         - "place radio on coffee table"

   actor:
     model:
       model_path: /path/to/pi05_base_openpi   # 新格式 fp32 基础权重
       openpi:
         task: sft
         embed_dtype: bf16
       openpi_data:
         norm_stats_path: /path/to/assets/behavior-1k/2025-challenge-demos/norm_stats.json

- ``use_skill: false`` 用主任务文本；``true`` 时按窗口从 ``task_subtasks`` 取逐帧技能文本（数据集里折叠后的 orchestrator 不能当标签）。
- ``fine_grained_level`` / ``tolerance_s`` 控制流式时间对齐。

.. code:: bash

   bash examples/sft/run_vla_sft.sh behavior_sft_openpi_pi05

Pi0 + RoboTwin
----------------------------------------

用官方 OpenPI / LeRobot map-style 加载器，``config_name: pi0_aloha_robotwin``。示例数据：`adjust_bottle <https://huggingface.co/datasets/RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data>`_；对应 HF SFT 权重：`RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle>`_。

.. code:: yaml

   data:
     train_data_paths: /path/to/robotwin-data
     num_workers: 4

   actor:
     model:
       model_path: /path/to/pi0_base_pytorch_new
       num_action_chunks: 50
       action_dim: 14
       openpi:
         task: sft
         embed_dtype: bf16
         config_name: "pi0_aloha_robotwin"
         num_images_in_input: 3
       openpi_data:
         norm_stats_path: ${actor.model.model_path}/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

14 维 ALOHA 动作、3 路图像；进模型前按 OpenPI 规则 pad 到 32 维。``openpi_data.norm_stats_path`` 把训练和评估钉到同一份 ``norm_stats.json``。换任务时改路径里的任务名。

.. code:: bash

   bash examples/sft/run_vla_sft.sh robotwin_adjust_bottle_sft_openpi


Pi0.5 + SpiritAI
--------------------

使用 ``spiritai_sft_openpi_pi05_full.yaml`` 在双卡上训练 Pi0.5 的全部参数。该 recipe 复用 RLinf 的 FSDP 和 flow-matching loss，通过已有登录配置向 W&B 记录指标。为每个任务指定数据集及对应的归一化统计。

如果全局 W&B CLI 与项目 venv 的默认服务地址不同，启动前将 ``WANDB_BASE_URL`` 设置为已有登录使用的服务地址。凭据按服务主机名匹配。

Cartesian 布局共 25 维：左臂位姿（6）、psi（1）、夹爪（1），右臂位姿（6）、psi（1）、夹爪（1），躯干位姿（6）和底盘速度（3）。三路相机为 ``cam_high``、``cam_left_wrist`` 和 ``cam_right_wrist``。当前 state 归一化后与 prompt 一起编码为 token；未来动作补齐到 32 维，horizon 为 50。256 的 token 上限同时容纳 prompt 和 state，较长任务指令需要检查截断。

使用当前任务自己的 stats 文件。对于固定底盘的 FoldBox 任务，设置 ``zero_base_action_targets=true``，在归一化之前将底盘 target 置零：

.. code-block:: bash

   export JAX_PLATFORMS=cpu
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   python examples/sft/train_vla_sft.py \
     --config-name spiritai_sft_openpi_pi05_full \
     data.train_data_paths=/path/to/foldbox_dataset \
     actor.model.model_path=/path/to/pi05_base_rlinf_pytorch \
     actor.model.openpi_data.norm_stats_path=/path/to/foldbox_assets/norm_stats.json \
     actor.model.openpi_data.zero_base_action_targets=true \
     runner.logger.log_path=./logs/spiritai_foldbox \
     runner.logger.experiment_name=spiritai_foldbox_pi05_full_bs32

该命令采用全局 batch 32、90,000 个 optimizer steps、FP32 参数更新、BF16 计算、梯度检查点和 ``full_shard``。``JAX_PLATFORMS=cpu`` 将数据工具限制到 CPU，PyTorch 仍在 GPU 上训练。持续运行前，检查 W&B 中的 loss、梯度范数、学习率和单步耗时。

测量峰值显存后再增加每卡 ``actor.micro_batch_size``。通过 ``actor.global_batch_size`` 独立比较 32、64 和 128。在样本曝光量相同的情况下，分别对应 90,000、45,000 和 22,500 个 optimizer steps。每组实验显式配置 ``runner.max_steps``、``actor.optim.total_training_steps``、warmup 和保存间隔。

双卡实测流程、梯度累积与 GPU 批量的区别，以及 NCCL 实际路径检查，见 :doc:`调优 PCIe 多卡 OpenPI SFT <../../guides/openpi_pcie_sft>`。

可选的 ``data.stateful_loader=true`` 通过现有 SFT checkpoint 接口保存已消费的数据位置和 worker 随机状态。该 loader 暴露有限 epoch，由 SFT worker 在每个 epoch 更新分布式 sampler。恢复时，在相同训练命令后添加 ``+runner.resume_dir=/path/to/checkpoints/global_step_5000``。保持数据内容、变换、stats、rank 布局、micro batch、worker 数和 seed 一致。数据集和 stats 应与实验记录一起保留，权重文件不会打包这些资产。

本 recipe 在 glibc 系统上还设置 ``data.worker_malloc_trim_interval=64``，由各数据子进程归还符合条件的空闲堆页，设为 0 可关闭。此分配器设置不改变数据恢复签名，持续内存检查见 :doc:`PCIe 调优指南 <../../guides/openpi_pcie_sft>`。

使用已有 SpiritAI container 部署训练结果，步骤见 :doc:`真机评测 <../../evaluations/guides/realworld>`。RLinf RTC 在推理时引导采样，本 recipe 训练普通 flow-matching 目标，不使用干净动作前缀训练或仅对后缀计算 loss。

训练显式子任务模型
----------------------------------------

对于 SpiritAI 折盒任务，使用 ``spiritai_sft_openpi_pi05_subtask_lora`` 在已有 Pi0.5 base 上微调两种监督视图。该 downstream recipe 采用 `Pi0.5 原论文 <https://arxiv.org/abs/2504.16054>`_ 的显式高低层分解，训练目标为 subtask 文本 CE 和连续动作 flow matching。

.. list-table:: 加载时构造的训练视图
   :header-rows: 1

   * - 分支
     - 输入
     - 目标
   * - 高层
     - 当前图像、归一化 state、overall instruction
     - 当前 subtask 文本及 EOS
   * - 低层
     - 当前图像、归一化 state、当前 subtask
     - 50 步动作窗口，补齐到 32 维

数据保留 LeRobot v2.1 布局。``task_index`` 通过 ``meta/tasks.jsonl`` 指向 overall instruction；每个 parquet 行另有非空 ``subtask`` 字符串，并在 ``meta/info.json.features`` 中声明。loader 保持逐帧对齐，在线构造两种视图。低层文本只使用 subtask；高层 response 使用因果 attention，仅对 subtask token 和 EOS 计算 CE。高层序列超长时直接报错，避免截断监督标签。

原生 LoRA 覆盖两个 Gemma expert 的 attention 和 FFN。VLM 的 rank/alpha 为 16/16，Action Expert 为 32/32，对应 `OpenPI Gemma <https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py>`_ 为两个 expert 设置的默认值。RLinf 的 :doc:`LoRA 指南 <../../guides/lora>` 给出通用 rank 32 示例，没有规定 Pi0.5 双专家联合 SFT 的专用配置。视觉编码器、共享的 token embedding/output 权重、动作与时间投影、norm 和其余 base 参数保持冻结。FM 梯度通过 Action Expert 的 attention 传到 VLM adapter。tokenizer 复用 PaliGemma 词表，高层前缀以 ``Subtask:`` 结束。

``ce_chunk_size`` 限制每次投影的 response token 数。投影和 CE 分块在 embedding/output 模块的一次 forward 内执行，使标签长度不同的 rank 以相同次数进入该 FSDP wrapper。扩展 loss 时须保留这一边界，不能让 rank 内的分块次数决定分片模块的调用次数。

不同长度触发的故障、修复及验证边界见 :ref:`openpi-subtask-collective-order`。

该固定底盘 FoldBox recipe 设置 ``zero_base_action_targets=true``，与已有全参数 FoldBox 运行及其归一化文件保持一致。

.. warning::

   预处理须与归一化统计对应。底盘动作 quantile 范围为零时，非零原始底盘指令会被放大。对于底盘动作统计均为零的情况，分层 loader 要求使用对应的 ``zero_base_action_targets=true`` 变换。

动作窗口沿用已有 sampler，可以跨越 subtask 边界，整个窗口以当前帧的标签为条件。

执行配置采用 FSDP1 ``full_shard``，设置 ``backward_prefetch=pre`` 和 ``gradient_checkpointing=false``。``openpi.torch_compile=true`` 编译现有 GELU-GLU kernel，不是编译整个模型。比较运行结果时，保留 recipe 的 FP32 参数更新、BF16 主干计算和 FP32 梯度归约策略。

先用当前任务的归一化统计完成 100 次更新的双卡检查。使用现有 venv 的 Python；两个 runner 覆盖项将检查限制为 100 次更新，并禁用全部 checkpoint 保存，包括最后一步：

.. code-block:: bash

   export JAX_PLATFORMS=cpu
   export PYTHONPATH="$PWD:${PYTHONPATH:-}"
   .venv/bin/python examples/sft/train_vla_sft.py \
     --config-name spiritai_sft_openpi_pi05_subtask_lora \
     data.train_data_paths=/path/to/20260805_FoldBox_SpiritAI_Moz1WB_10subtasks \
     actor.model.model_path=/path/to/pi05_base_rlinf_pytorch \
     actor.model.openpi_data.norm_stats_path=/path/to/foldbox_assets/norm_stats.json \
     runner.max_steps=100 runner.save_interval=-1 \
     runner.logger.log_path=./logs/spiritai_subtask_check \
     runner.logger.experiment_name=spiritai_pi05_subtask_ce_flow_lora_100

该命令使用两个 FSDP rank，每卡 micro batch 8，全局 batch 64，累积 4 轮梯度。每次 optimizer update 包含 32 条高层记录和 32 个低层动作窗口，两个 rank 按相同顺序交替执行分支。每种 loss 在各自样本上取平均，总目标为 ``text_loss_weight * CE + flow_loss_weight * FM``。初始权重为 1 和 10，只作为起始超参数，不代表已经验证折盒性能提升。观察 ``subtask_ce``、``flow_loss``、``hl_records_per_update``、``ll_action_chunks_per_update``、``hl_tokens_per_update`` 和 ``peak_allocated_gib``；与 flow-only 训练比较时，同时记录动作窗口曝光量和 optimizer steps。

``time/training`` 测量完整 optimizer update，包含全部 4 轮累积、数据等待、计算和训练通信，不是单个 microbatch 耗时。此前全参数 64/8 配置每次更新累积 4 轮、训练 64 个动作窗口；当前为 2 轮 HL、2 轮 LL，共 32 个动作窗口。文本分支新增 257152 类词表投影和 CE。冻结的底座权重仍参与计算和 FSDP 参数收集；FSDP1 wrapper 混合冻结权重与 adapter 时，还可能为整个 flat parameter 分配和归约梯度。:ref:`openpi-joint-update-timing` 记录的是此前 32/2 运行的耗时及其边界，不能作为 64/8 的性能验证。

检查期间保留 45000 次更新的 scheduler 计划和 750 次更新的 warmup，因此 100 次更新全部处于 warmup。通过检查只能验证训练链路和指标有限，不能证明收敛或 warmup 后的稳定性。正式训练时移除 ``runner.max_steps=100`` 和 ``runner.save_interval=-1``，并使用新的日志目录和实验名。recipe 随后运行 45000 次更新，每 5000 次保存。AdamW 的学习率和最小学习率都为 ``1e-5``，所以 ``openpi_cosine`` 在 warmup 后保持恒定学习率。

保持 ``stateful_loader=true``，梯度累积轮数须为不小于 2 的偶数。添加 ``+runner.resume_dir=/path/to/checkpoints/global_step_5000``，恢复 optimizer、scheduler、loader 分支位置和 RNG 状态。恢复时，数据内容、预处理、归一化、rank 布局、micro batch、worker 数、seed 和分层训练设置须保持一致。

训练完成后，先合并原生 adapter，再部署：

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.merge_lora \
     --ckpt /path/to/checkpoints/global_step_5000 \
     --output /path/to/merged_pi05_subtask_lora

导出目录包含裸 ``full_weights.pt``、``norm_stats.json`` 和记录训练设置及文本模板版本的 ``openpi_training.json``。评估使用 ``realworld_spiritai_eval_pi05_subtask_RTC``，将模型和归一化路径指向该目录，保持训练时的 token 上限和 state 编码约定，设置 ``openpi.task=eval``。将未合并的 adapter 权重当作裸 policy 加载时会报错。

推理先生成 subtask，再以当前 observation 和该文本重建低层上下文，高层 KV cache 与动作 cache 独立。直接调用 policy 时每次重新规划；保留 ``SubtaskPlanningState`` 的调用方须在 episode 边界清空它。RTC worker 持有该状态，收到 ``bootstrap`` 时清空，``planning_interval=3`` 表示每 3 次动作请求刷新高层。subtask 变化时丢弃上一动作 chunk 的 guidance。日志记录生成文本和规划耗时；空 subtask 会阻止动作生成。训练与推理链路验证后，再独立评估真机成功率及跨盒型号泛化。

评测
----------------------------------------

以下直接加载流程适用于 flow-only checkpoint；显式 subtask LoRA 使用上文的合并导出流程。

SFT 会把 ``full_weights.pt`` 写到 ``.../checkpoints/global_step_<N>/actor/model_state_dict/``。``openpi`` 评测可以直接加载这个 ``.pt``\ （或 ``global_step_<N>`` 目录）：加载时会剥掉 FSDP / wrapper 前缀，模型形状来自评测 YAML，不必先转成 ``model.safetensors``。

不要把训练的 ``openpi.task: sft`` 带到评测里——评测配置必须是 ``task: eval``\ （才会走 ``Pi0Eval`` 和观测 transform）。``config_name``、``num_action_chunks`` 与训练对齐。

归一化统计也\ **不会**\ 跟着 ``full_weights.pt`` 走。现成评测 YAML 常把 ``openpi_data.norm_stats_path`` 插值成 ``${rollout.model.model_path}/.../norm_stats.json``，那是给「权重和统计在同一目录」的转换产物用的。只改 ``model_path`` 指到 SFT checkpoint 时，必须\ **另外**\ 把统计指回训练用的那份（RoboTwin：基础权重旁的 ``physical-intelligence/robotwin/<task>/norm_stats.json``；BEHAVIOR：训练时的 ``openpi_data.norm_stats_path``）。

.. code:: bash

   # 示例：直接评测 SFT 产出的 .pt（统计仍用训练时的文件，不要跟 model_path 走）
   bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
     rollout.model.model_path=/path/to/logs/.../checkpoints/global_step_30000 \
     rollout.model.openpi_data.norm_stats_path=/path/to/pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

评测步骤见 :doc:`BEHAVIOR-1K <../../evaluations/guides/behavior>` 和 :doc:`RoboTwin <../../evaluations/guides/robotwin>`。如果要导出可分发的裸 ``Pi0`` 目录（``model.safetensors`` + ``config.json`` + 拷进去的 ``norm_stats.json``），用 ``sft_to_openpi``，命令见 ``rlinf/utils/ckpt_convertor/openpi/README.md``。

可视化
----------------------------------------

看 **训练损失** 是否在下降。指标含义见 :doc:`训练指标 <../../reference/metrics>`。

.. code:: bash

   tensorboard --logdir ./logs
