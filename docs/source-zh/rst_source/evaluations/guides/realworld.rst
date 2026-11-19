真机评测
========

在 Franka 机械臂或已有 SpiritAI 机器人 container 上部署 VLA policy。Franka 部分介绍本地硬件配置，SpiritAI 部分介绍 GPU 工作站与机器人服务器的连接。

相关训练文档：:doc:`../../examples/embodied/franka_pi0_sft_deploy`、:doc:`../../examples/embodied/franka`、:doc:`../../examples/embodied/sft_dreamzero`

环境准备
--------

**硬件**

- Franka Emika Panda 机械臂 + Intel RealSense 相机（默认）
- 一台 **GPU 节点** （训练 / rollout）与一台 **机器人控制节点** （直连 Franka 与相机，无需 GPU）
- 所有节点处于同一局域网；机械臂仅需与控制节点互通

使用 ZED 相机或 Robotiq 夹爪时，请参阅 :doc:`../../examples/embodied/franka_zed_robotiq`。

**依赖安装**

控制节点与 GPU 节点需分别安装依赖：

.. code-block:: bash

   # 机器人控制节点（Franka + 相机）
   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

.. code-block:: bash

   # GPU / rollout 节点（π₀ 评测）
   bash requirements/install.sh embodied --model openpi --env franka
   source .venv/bin/activate

控制节点使用默认的 Franky backend。如需改用 ROS 控制机械臂，请参考 :doc:`../../examples/embodied/franka` 的「旧版 ROS 后端（可选）」章节。

DreamZero 真机评测还需在 GPU 节点安装 DreamZero 依赖，详见 :doc:`../../examples/embodied/sft_dreamzero`。

**节点拓扑**

真机评测通常采用 **「1 个 GPU 节点 + 1 个 Franka 控制节点」** 的异构布局：

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``RLINF_NODE_RANK``
     - 角色
     - 说明
   * - rank 0
     - GPU / head
     - 运行 ``rollout``；在此节点提交 ``run_eval.sh``
   * - rank 1
     - 机器人控制
     - 运行 ``env`` worker，直连 Franka 与相机

``realworld_pnp_eval.yaml`` 与 ``realworld_pnp_eval_dreamzero.yaml`` 使用上述双节点布局；``realworld_eval.yaml`` （自定义任务）为 **单机** 布局，``env`` 与 ``rollout`` 均部署在同一 Franka 节点。

Ray 集群的完整搭建、固件版本与 libfranka 兼容性见 :doc:`../../examples/embodied/franka` 与 :doc:`../../guides/realworld_robot`。

示例配置
--------

``evaluations/realworld/`` 目录下已有以下示例：

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - 配置文件
     - 任务
     - 模型
   * - ``realworld_pnp_eval.yaml``
     - Bin-relocation (PnP)
     - π₀
   * - ``realworld_pnp_eval_dreamzero.yaml``
     - Bin-relocation (PnP)
     - DreamZero
   * - ``realworld_eval.yaml``
     - 自定义任务（``FrankaEnv-v1``）
     - π₀

若 ``evaluations/realworld/<config>.yaml`` 不存在，``run_eval.sh`` 会回退到 ``examples/embodiment/config/`` 下同名配置（需设置 ``runner.task_type: embodied_eval`` 与 ``runner.only_eval: True``）。详见 :doc:`../reference/cli`。
双节点 Dual Franka 部署通过该回退路径使用 ``realworld_eval_dual_franka``。单机 RTC 评测使用 ``evaluations/realworld/realworld_dual_franka_pi05_RTC.yaml``。

启动前检查
----------

在正式评测前，建议依次完成以下检查：

1. **相机连接** （控制节点）：

   .. code-block:: bash

      python -m toolkits.realworld_check.test_franka_camera

   记录输出的相机序列号，填入 ``env.eval.override_cfg.camera_serials``。

2. **目标位姿** （PnP 任务，控制节点）：

   .. code-block:: bash

      export FRANKA_ROBOT_IP=<robot_ip>
      python -m toolkits.realworld_check.test_franka_controller
      # 输入 getpos_euler 获取目标末端位姿

   将结果填入 ``env.eval.override_cfg.target_ee_pose``。PnP 任务中该位姿表示运动空间中间的最低点，并用于成功判定与工作空间截断；详见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`。

3. **Ray 集群连通** （各节点）：

   .. code-block:: bash

      ray status

   应同时看到 GPU 节点与 Franka 节点在线。

4. **Dummy 模式（可选）**：参考 ``examples/embodiment/config/realworld_dummy_franka_sac_cnn.yaml``，在 ``override_cfg`` 中设置 ``is_dummy: True`` 验证集群配置，无需真实机器人运动。

.. warning::

   真机评测前务必确认工作空间限位（``ee_pose_limit_min`` / ``ee_pose_limit_max``）与急停功能正常；首次评测建议将 ``env.eval.rollout_epoch`` 设为较小值。

Ray 集群启动
------------

在 **每个节点**、执行 ``ray start`` **之前**，先统一环境变量（可使用 ``ray_utils/realworld/setup_before_ray.sh``）：

.. code-block:: bash

   source ray_utils/realworld/setup_before_ray.sh
   export RLINF_NODE_RANK=<0|1>          # 集群内唯一
   # 多网卡时指定对外可达网卡：
   # export RLINF_COMM_NET_DEVICES=<network_interface>

若 setup 脚本未激活 Franka 环境，控制节点还需手动激活：

.. code-block:: bash

   source .venv/bin/activate

然后启动 Ray（记 head 节点 IP 为 ``<head_ip>``）：

.. code-block:: bash

   # GPU 节点（rank 0，head）
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<head_ip>

   # Franka 控制节点（rank 1）
   export RLINF_NODE_RANK=1
   ray start --address=<head_ip>:6379

.. important::

   ``ray start`` 会冻结当时的 Python 解释器与环境变量；请在各节点完成 venv、``PYTHONPATH`` 等配置后再启动 Ray。

完整评测流程（PnP / π₀）
------------------------

**Step 1：配置 Ray 集群**

按上文在各节点启动 Ray，确保 ``ray status`` 显示双节点在线。

**Step 2：准备模型**

π₀ PnP 评测需要：

- ``rollout.model.model_path``：Pi0 基座模型目录，且须包含 ``<repo_id>/norm_stats.json`` （SFT 前置步骤生成，见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`）
- ``runner.ckpt_path``：SFT 导出的 ``full_weights.pt``

**Step 3：编辑配置**

修改 ``evaluations/realworld/realworld_pnp_eval.yaml``，至少更新以下字段：

.. code-block:: yaml

   cluster:
     node_groups:
       - label: "4090"
         node_ranks: 0          # GPU 节点 rank
       - label: franka
         node_ranks: 1          # Franka 控制节点 rank
         hardware:
           type: Franka
           configs:
             - robot_ip: ROBOT_IP
               node_rank: 1

   runner:
     ckpt_path: /path/to/full_weights.pt

   env:
     eval:
       rollout_epoch: 20
       override_cfg:
         target_ee_pose: [0.50, 0.00, 0.01, 3.14, 0.0, 0.0]
         camera_serials: ["CAMERA_SERIAL_1", "CAMERA_SERIAL_2"]
         task_description: "pick up the object and place it into the container"

   rollout:
     model:
       model_path: /path/to/pi0-model
       openpi:
         config_name: "pi0_realworld"

``node_ranks`` 与 ``component_placement`` 须与集群中实际的 ``RLINF_NODE_RANK`` 一致。

**Step 4：启动评测**

在 **GPU / head 节点** 执行：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_pnp_eval

也可通过 Hydra 覆盖参数，无需修改 YAML 文件：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_pnp_eval \
     rollout.model.model_path=/path/to/pi0-model \
     runner.ckpt_path=/path/to/full_weights.pt

**Step 5：查看结果**

策略以 ``runner.only_eval: True`` 模式运行；终端输出任务指标，日志见 :doc:`../reference/results`。

.. _realworld-eval-config:

评测配置详解
------------

以下说明适用于 ``realworld_pnp_eval.yaml``；自定义任务评测见下文，DreamZero 见 :doc:`../../examples/embodied/sft_dreamzero`。

必须修改的字段
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 字段
     - 位置
     - 说明
   * - ``robot_ip``
     - ``cluster.node_groups[].hardware.configs``
     - Franka 机器人 IP
   * - ``node_ranks`` / ``component_placement``
     - ``cluster``
     - 与 ``RLINF_NODE_RANK`` 对齐；env 在 Franka 节点，rollout 在 GPU 节点
   * - ``target_ee_pose``
     - ``env.eval.override_cfg``
     - PnP 目标末端位姿 ``[x,y,z,rx,ry,rz]``，影响成功判定与运动截断
   * - ``camera_serials``
     - ``env.eval.override_cfg``
     - RealSense 序列号列表（**非** ``node_groups`` 字段）
   * - ``task_description``
     - ``env.eval.override_cfg``
     - 语言指令，须与 SFT 训练一致
   * - ``model_path``
     - ``rollout.model``
     - Pi0 基座模型目录（含 ``norm_stats.json``）
   * - ``ckpt_path``
     - ``runner``
     - SFT checkpoint（``full_weights.pt``）
   * - ``config_name``
     - ``rollout.model.openpi``
     - PnP 任务为 ``"pi0_realworld"``

关键 ``env.eval`` 字段
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 字段
     - 说明
   * - ``rollout_epoch``
     - 评测轮数，每轮重复完整 rollout；默认 20
   * - ``max_episode_steps``
     - 单条轨迹最大步数，默认 200
   * - ``max_steps_per_rollout_epoch``
     - 每轮 rollout 步数上限；**必须能被** ``rollout.model.num_action_chunks`` **整除** （PnP 默认 ``num_action_chunks=10``）
   * - ``total_num_envs``
     - 并行环境数，真机通常为 1
   * - ``teleop``
     - 从策略手中接管的设备，评测时通常填 ``none``

``run_eval.sh`` 行为
~~~~~~~~~~~~~~~~~~~~

- ``realworld`` benchmark **不** 调用 ``setup_sim_env``，无需仿真相关环境变量
- 日志写入 ``logs/<时间戳>-<config_name>/eval_embodiment.log``
- 评测入口脚本须在 **head（GPU）节点** 提交

DreamZero 真机评测（``realworld_pnp_eval_dreamzero.yaml``）完整流程见 :doc:`../../examples/embodied/sft_dreamzero`。

自定义真机任务评测
------------------

本节面向 **你自己定义的任务**，与上文内置的 Bin-relocation（PnP）任务不同。

RLinf 提供通用真机环境 ``FrankaEnv-v1``：在 YAML 中配置 ``task_description``、目标/复位位姿、工作空间限位等即可，**无需编写新的 Python 环境类**。典型用法是：采集示教数据 → 监督微调（SFT）训练策略 → 在真机上部署评测（流程见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`）。环境模板为 ``examples/embodiment/config/env/realworld_franka_sft_env.yaml``；评测配置为 ``evaluations/realworld/realworld_eval.yaml``。

**节点拓扑**

``realworld_eval.yaml`` 为 **单机** 配置：``env`` 与 ``rollout`` 均部署在 Franka 节点（``num_nodes: 1``），适用于 GPU 与控制合署一机的场景。

关键 ``override_cfg`` 字段
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   override_cfg:
     task_description: "pick up the object and place it into the container"
     target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]   # 目标位姿
     reset_ee_pose:  [0.5, 0.0, 0.2, -3.14, 0.0, 0.0]    # 复位位姿（应高于目标）
     max_num_steps: 300
     reward_threshold: [0.01, 0.01, 0.01, 0.2, 0.2, 0.2]
     action_scale: [1.0, 1.0, 1.0]
     ee_pose_limit_min: [0.4, -0.2, 0.05, -3.64, -0.5, -0.5]
     ee_pose_limit_max: [0.6,  0.2, 0.35, -2.64,  0.5,  0.5]

**启动评测**

替换 ``ROBOT_IP`` 与 ``MODEL_PATH`` 后执行：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_eval

自定义任务的采集、训练与部署流程见 :doc:`../../examples/embodied/franka_pi0_sft_deploy`。

Dual Franka 部署
----------------

双臂评测有两份配置。

``examples/embodiment/config/realworld_eval_dual_franka.yaml`` 是双节点回退配置。``run_eval.sh`` 在 ``evaluations/realworld/`` 找不到同名文件时使用它。将 ``rollout.model.model_path`` 设为已同步的 checkpoint 目录，将 ``actor.model.openpi_data.repo_id`` 设为包含 ``norm_stats.json`` 的 repo id。

``evaluations/realworld/realworld_dual_franka_pi05_RTC.yaml`` 是单机 RTC 评测：GPU、两台机械臂和三路相机在同一台机器上。动作是 20 维 tcp_rot6d，``openpi.config_name`` 为 ``pi05_dualfranka_tcp_rot6d``，主相机是 ``left_wrist_0_rgb``。把 ``MODEL_PATH``、``norm_stats`` 路径、机械臂 IP、相机 serial、夹爪串口、``keyboard_device`` 和 ``TASK_DESCRIPTION`` 换成自己的值。复位完成后，在这台 USB 键盘或脚踏上按 ``a`` 才开始 policy；``c`` 记成功，``b`` 记失败。脚本不读终端输入。``a`` 要按在 ``keyboard_device`` 指向的 evdev 设备上，该字段为空时用 ``RLINF_KEYBOARD_DEVICE``。SSH 会话里输入的 ``a`` 到不了这台设备；如果这台设备就是本机键盘，即使焦点在终端上，按下也会生效。无图形界面时保持 ``enable_camera_player: False``。``max_steps_per_rollout_epoch`` 必须能被 20 整除。

.. code-block:: bash

   bash evaluations/run_eval.sh realworld_eval_dual_franka \
       rollout.model.model_path=/path/to/deploy/global_step_<N> \
       actor.model.openpi_data.repo_id=<repo_id>/tcp_rot6d_v1 \
       env.eval.override_cfg.task_description="handover the object"

   bash evaluations/run_eval.sh realworld_dual_franka_pi05_RTC

完整采集、SFT、checkpoint 同步与按键流程见 :doc:`../../examples/embodied/dual_franka` 和 :doc:`../../examples/embodied/dual_franka_pico_dagger`。

查看结果
--------

- **终端指标：** 任务成功率、回报等（具体字段因环境而异）
- **日志：** ``logs/<时间戳>-<config_name>/eval_embodiment.log``
- **视频：** 若 ``env.eval.video_cfg.save_video: True``，视频保存在 ``video_base_dir`` 或 ``<log_path>/video/eval/``

详见 :doc:`../reference/results`。

常见问题
--------

- **安全：** 评测前确认工作空间限位与急停功能正常；首次评测降低 ``rollout_epoch``。
- **节点拓扑：** ``env`` worker 必须部署在可直连 Franka 的节点；``node_ranks`` 须与 ``RLINF_NODE_RANK`` 一致。PnP 与 ``realworld_eval_dual_franka`` 为双节点；自定义任务评测和 ``realworld_dual_franka_pi05_RTC`` 为单机。
- **相机未找到：** 在控制节点运行 ``python -m toolkits.realworld_check.test_franka_camera``，核对 ``camera_serials``。
- **动作异常：** 检查 ``norm_stats.json`` 是否位于 ``model_path/<repo_id>/``，以及 ``openpi.config_name`` 是否与训练一致。
- **Ray 只见一个节点：** 检查防火墙、``RLINF_COMM_NET_DEVICES`` 与 head IP 是否可被其他节点访问。
- **步数报错：** 确保 ``max_steps_per_rollout_epoch`` 能被 ``num_action_chunks`` 整除。
- **ZED / Robotiq：** 参阅 :doc:`../../examples/embodied/franka_zed_robotiq`。

SpiritAI 机器人 Container
--------------------------

通过已有机器人 WebSocket container 部署 Full Flow 或 Explicit Subtask LoRA，每次选择一种 policy。一个 rollout worker 在一张 GPU 上持有模型，CPU env worker 独占机器人连接，两套配置共用原生 evaluation runner 与 RTC workers。机器人 container 负责相机和 SDK 执行；``runtime_setup=false`` 保留本机 ROS 进程。

.. list-table:: 选择 Policy
   :header-rows: 1
   :widths: 35 65

   * - 配置
     - 推理契约
   * - ``realworld_spiritai_eval_pi05_full_deploy``
     - 观测 + overall 指令 → Flow 动作，关闭 hierarchy。
   * - ``realworld_spiritai_eval_pi05_subtask_deploy``
     - 观测 + overall → 生成 Subtask；观测 + Subtask → Flow 动作。先合并 LoRA adapter。

准备 Checkpoint 记录
~~~~~~~~~~~~~~~~~~~~

使用实际 W&B ``config.yaml`` 和 ``wandb-metadata.json`` 生成新的部署目录。Full Flow 还会根据所有 rank 保存的 ``actor/data.pt`` loader signature 核对 normalization 指纹；Subtask 使用合并 checkpoint 中的 ``openpi_training.json``。替换路径，并将 ``POLICY_KIND`` 设为 ``flow_only`` 或 ``explicit_subtask``：

.. code-block:: bash

   .venv/bin/python -m rlinf.utils.ckpt_convertor.openpi.prepare_deployment \
     --policy-kind POLICY_KIND \
     --ckpt /path/to/native_or_merged_checkpoint \
     --training-config /path/to/wandb/run/files/config.yaml \
     --launch-metadata /path/to/wandb/run/files/wandb-metadata.json \
     --output /path/to/new_deployment_bundle

该命令写入 ``openpi_deployment.json`` 和 ``norm_stats.json``，保留源权重与历史训练 metadata。manifest 保存训练设置、来源 hash、step、模式和完整权重文件 SHA256；启动时核对权重、normalization、state/token 设置与模型维度。大 checkpoint 的 hash 会增加启动 I/O。记录到的 Git HEAD 不能证明历史工作区干净。

准备 Full Flow 时，保留原生 ``global_step_<N>/actor/model_state_dict/full_weights.pt`` 和 ``actor/data.pt`` 布局。Subtask LoRA 先使用已有 ``merge_lora`` 导出器，再对合并目录执行准备命令。工具只读取可信本地 checkpoint，拒绝覆盖已有输出目录，也拒绝未合并 adapter。

只读检查 Container
~~~~~~~~~~~~~~~~~~

启动 bridge 前，读取 metadata、状态、三路 RGB 图像和 25D 测量状态：

.. code-block:: bash

   .venv/bin/python -m toolkits.realworld_check.check_spiritai \
     --url ws://ROBOT_HOST:8766 --require-execution

检查只发送 ``get_status`` 和 ``get_obs``，随后释放独占连接；执行能力检查不会开启 following。whole-body profile 要求 Cartesian endpoint 至少为 20D，以保留 torso 指令；23D 还包含固定为零的 base 指令。client 从 25D policy 动作中删除两个 psi。

``execution_capabilities`` 下的 ``single_step_cartesian``、``tracked_completion`` 和 ``cancel_actions`` 是当前 RLinf 适配新增的约定，原 OpenPI bridge 没有要求这些字段。缺少声明不能单独证明缺少行为，更新前应核对实际 server 源码。兼容实现让单行目标绕过多点插值，busy 跟踪后台 SDK 下发 Future，``get_status.execution_error`` 暴露 ACK 后的错误；取消在后续 Python 下发停止后回复。Future 完成或取消均不表示机械到位，也不撤销控制器已接受的目标；回复携带对应 ``request_id``。

对于已支持的本地 server 版本，生成可审阅的兼容副本：

.. code-block:: bash

   .venv/bin/python -m toolkits.realworld_check.prepare_spiritai_server \
     --source-root /path/to/robot_server_code \
     --output /path/to/new_reviewed_server_copy

工具核对源文件与补丁后的精确 hash，将归档补丁应用到新副本，保留原源码，不安装或启动副本。需另外核对机器人运行中的 container 版本并部署匹配代码；能力标志本身不能证明执行行为正确。

预览所选 Policy
~~~~~~~~~~~~~~~

先使用默认观测模式，指定所选配置、checkpoint、已准备的 bundle 和机器人地址：

.. code-block:: bash

   PATH="$PWD/.venv/bin:$PATH" bash evaluations/run_eval.sh realworld \
     realworld_spiritai_eval_pi05_full_deploy \
     cluster.component_placement.rollout=0 \
     rollout.model.model_path=/path/to/checkpoints/global_step_45000 \
     rollout.model.openpi.deployment_metadata_path=/path/to/deployment/openpi_deployment.json \
     rollout.model.openpi_data.norm_stats_path=/path/to/deployment/norm_stats.json \
     env.eval.override_cfg.robot_url=ws://ROBOT_HOST:8766 \
     env.eval.max_steps_per_rollout_epoch=300 \
     env.eval.max_episode_steps=300 \
     env.eval.override_cfg.execution_enabled=false \
     'runner.logger.logger_backends=[]'

该命令在 ``execution_enabled=false`` 下运行 policy，保持动作和 following 状态不变。用 ``cluster.component_placement.rollout`` 指定一个 RLinf GPU hardware rank。切换 Subtask LoRA 时，改用 ``realworld_spiritai_eval_pi05_subtask_deploy``，并传入它的合并 checkpoint 和对应 bundle。两者分开运行，每次使用一张 GPU 和同一套 bridge。

这三个 env 覆盖值是 Hydra 运行配置，不是模型 observation。YAML 默认是 ``3000 / 3000 / false``，命令行优先；接受默认值时可以省略，也可以在 ``realworld_spiritai_eval_pi05_full_deploy.yaml`` 中持久化 Full 的设置。两个长度参数由不同组件使用：

.. list-table:: 运行配置覆盖
   :header-rows: 1
   :widths: 42 58

   * - 参数
     - 作用
   * - ``env.eval.max_steps_per_rollout_epoch``
     - worker 每轮控制循环预算，当前 validator 要求可被 action horizon 50 整除。
   * - ``env.eval.max_episode_steps``
     - 环境单 episode 上限，到达时产生 timeout truncation/done；不要求与 worker 预算相等。
   * - ``env.eval.override_cfg.execution_enabled``
     - false 仅预测/读取；true 在预热和 bootstrap 后提交机器人动作。

单轮、auto-reset 关闭时，通常先到的上限结束本轮；一起设成目标 tick 数可保持边界一致。tick 是控制步，不是模型查询次数、Euler 次数或训练更新。延长运行无需修改 ``source_hz`` 或模型 horizon。

两套配置均使用三路 320×240 RGB 图像、25D Cartesian state、state token、256-token context、50 行 action chunk、10 次 Euler integration、BF16 与既有 FP32 islands，以及标称 30 Hz 控制。container/controller 的 120 Hz 是另一层频率。overall 和 normalization 应与任务一致。Full Flow 保留既定 overall-only 推理 baseline，即使历史训练 prompt 包含 current step；Subtask LoRA 自行生成文本，``planning_interval=3`` 表示三个 action request。每个 episode 重置 planner，Subtask 变化后停用上一 chunk 的 guidance。

开启执行与检查退出
~~~~~~~~~~~~~~~~~~

核对相机方向、位姿参考系、夹爪单位、起始构型和 container 行为后，在预览命令中追加 ``env.eval.override_cfg.execution_enabled=true``。profile 在预热和 bootstrap 后直接执行，不含键盘启动 gate。第一条有效指令下发前开启 following；平移、SO(3) 旋转与夹爪速度基于测量状态限幅，base 固定为零。示范夹爪范围 ``[0, 0.1]`` 仍需验证硬件契约。

Full Flow 约一分钟动作可在 preflight 后把两个长度参数设为 ``1500``，并开启执行。2026-11-19 的真机运行完成 1500 tick，首末 step 间隔 58.97 秒，约 25.42 Hz，无 abort 或关闭错误。这是依据实测频率换算，不是严格墙钟计时；模型加载和预热另计。操作人员确认机器人运动及推理链路正常，尚不构成量化折盒成功率。完整命令、参数来源、部署过程与 trace 证据保存在仓库 ``docs/experiments/spiritai-inference-bridge/``。

warmup 覆盖普通 Flow 和 ``approx`` 采样。``bootstrap_timeout_s=30`` 限制每个 warmup/bootstrap 请求，``request_timeout_s=1.2`` 限制 replan。应在实际 GPU 和网络上测量预算，包括 HL cache miss。保持 ``runner.rtc.enabled=true`` 使用异步调度；``rollout.model.openpi.rtc_enabled=false`` 只关闭 sampler guidance，保留同一个控制器。两种模式均报告真实 guidance 使用情况，共用行为见 :doc:`../../guides/rtc`。

env 同时只保留一个推理请求，拒绝迟到或错配回复，按本地 monotonic time 推进动作索引。warmup/bootstrap 失败时关闭连接，不发送 action 或 hold；完成过控制 tick 后，异常最多尝试两次实测位姿 hold，关闭连接并报告 ``rtc_aborted=1``。即使执行报错，关闭流程仍尝试释放 following；runner 在 worker 收尾后报告失败。

bounded evaluation 期间，driver 收到 SIGINT、SIGTERM 或 worker 失败通知后，通过 RLinf Channel 请求 env 停止，再退出；等待清理的上限为 ``shutdown_timeout_s``。最后一个未完成预测会被丢弃，不延迟连接关闭。进程或节点硬故障仍依赖 container 的断连取消。取消只停止后续 SDK 下发，不等价电机急停，也不撤销控制器最后的目标。

通过 ``<log_path>/<experiment_name>/rtc-env-0.jsonl`` 查看 session/chunk 标识、生成文本/EOS、真实 guidance、请求/推理延迟、预测与限幅指令、测量状态、hold 尝试及 ``closed`` 或 ``close_error``。YAML 保留 W&B，而本页预览命令关闭该可选后端，保留终端和 RTC JSONL。需要 W&B 时删除覆盖，并设置与已有凭据主机一致的 ``WANDB_BASE_URL``。``MetricLogger → wandb.init`` 报错发生在 worker 初始化及权重加载前。Full Flow 真机记录验证推理链路，物理任务成功率仍需单独评估。
