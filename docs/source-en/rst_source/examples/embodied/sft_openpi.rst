OpenPI Supervised Fine-Tuning
=============================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ vision-language-action models.

Run **full-parameter** or **LoRA** SFT on π₀ / π₀.₅ in RLinf. Policies use
``model_type: openpi``, a PyTorch port aligned with the JAX reference
(the official OpenPI PyTorch code is not). SFT is the usual RL cold start:
imitate demonstrations first, then continue with RL from a stronger prior.

Recipes
-------

Each experiment YAML imports a path-free model template through Hydra
``defaults``, then overrides paths and task fields. Replace ``/path/to/...``
before launch.

.. list-table::
   :header-rows: 1
   :widths: 18 38 22 22

   * - Task
     - Experiment config (``examples/sft/config/``)
     - Model template
     - Launch name
   * - LIBERO π₀
     - ``libero_sft_openpi.yaml``
     - ``model/pi0``
     - ``libero_sft_openpi``
   * - Real-world Franka bin-relocation
     - ``realworld_bin_relocation_sft_openpi.yaml``
     - ``model/pi0``
     - ``realworld_bin_relocation_sft_openpi``
   * - Custom LeRobot
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
   * - SpiritAI FoldBox explicit subtask
     - ``spiritai_sft_openpi_pi05_subtask_lora.yaml``
     - ``model/pi0_5``
     - ``spiritai_sft_openpi_pi05_subtask_lora``

For dual-Franka SFT and deployment, see :doc:`dual_franka_openpi_pytorch`.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: Methods
      :text-align: center

      Full SFT · LoRA

   .. grid-item-card:: Data
      :text-align: center

      LeRobot · BEHAVIOR

   .. grid-item-card:: Hardware
      :text-align: center

      1+ nodes · GPUs

| **You'll do:** install → prepare data and norm stats → ``bash examples/sft/run_vla_sft.sh <launch-name>`` → watch the training loss.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a matching dataset.

Shared conventions
------------------

``openpi.task`` must be ``sft`` during training. Eval YAMLs use ``eval``;
do not copy the training value.

``num_action_chunks`` is the env-executed chunk, **not** the network horizon:

- ``num_action_chunks`` / ``openpi.action_chunk``: how many actions the env
  executes per step.
- Network ``action_horizon``: ``openpi.action_horizon`` when set; otherwise the
  official ``TrainConfig.model.action_horizon`` for ``openpi.config_name``.
  Default templates only interpolate ``num_action_chunks`` into
  ``action_chunk``.

Official LeRobot SFT also sizes its data window from
``TrainConfig.model.action_horizon``; it does not read ``num_action_chunks``.
Typical official horizons: BEHAVIOR ``pi05_behavior`` **32**, RoboTwin
``pi0_aloha_robotwin`` / LIBERO ``pi0_libero`` **50**, real-world
``pi0_realworld`` **10**. Override ``openpi.action_horizon`` only when the
checkpoint horizon differs from that ``TrainConfig``.

The BEHAVIOR / dual-Franka streaming SFT loaders currently still use
``num_action_chunks`` as the data window, so those recipes keep it equal to
the official horizon.

Keep original parameters and optimizer updates in FP32 with
``actor.model.precision: fp32``. Set ``actor.model.openpi.embed_dtype: bf16``
for BF16 backbone activations and residuals. If omitted or null,
``embed_dtype`` follows ``model.precision``; ``precision: null`` selects BF16
activations. Configure FSDP separately: BF16 parameters
for forward/backward computation, FP32 gradient reduction, and FP32 buffers.
The shared model templates use ``precision: null`` for OpenPI's selective
BF16/FP32 parameter dtypes. BEHAVIOR, RoboTwin, dual-Franka, and RLT Stage 1
SFT recipes set ``precision: fp32``.

Select ``hybrid_engines/openpi_fsdp`` to keep projections, RMSNorm weights,
and vision embeddings in FP32. The profile preserves FP32 action/time inputs
and adaptive-norm conditioning while the backbone computes in BF16.
RLT Stage 1 recipes also keep ``rlt_module`` in FP32.

OpenPI supports FSDP1 ``no_shard``, ``shard_grad_op``, and ``full_shard`` with
``use_orig_params: true``. ``no_shard`` replicates all training state;
``shard_grad_op`` shards gradients and optimizer state; ``full_shard`` also
shards parameters between module executions. Select either sharded mode to
reduce per-rank memory. Enable ``gradient_checkpointing: true`` with
``gradient_checkpointing_use_reentrant: false`` to reduce activation memory.
BEHAVIOR / RoboTwin recipes use the ``openpi_cosine`` learning rate schedule.

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

The tokenizer is loaded by OpenPI's ``ModelTransformFactory`` from the base
model; the SFT YAML does not need a SentencePiece path.

Datasets
--------

Select the data format with ``actor.model.openpi.config_name``:

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - ``config_name``
     - Dataset / environment
   * - ``pi0_libero`` · ``pi05_libero``
     - LIBERO
   * - ``pi0_realworld``
     - Real-world Franka
   * - ``pi05_yam_joint``
     - Real-world Dual YAM
   * - ``pi0_aloha_robotwin``
     - RoboTwin (ALOHA)
   * - ``pi05_behavior``
     - BEHAVIOR-1K (streaming loader)
   * - ``pi0_maniskill`` · ``pi05_maniskill``
     - ManiSkill
   * - ``pi05_metaworld``
     - MetaWorld
   * - ``pi05_calvin``
     - CALVIN
   * - ``pi0_custom``
     - Custom LeRobot

Custom LeRobot dataset
~~~~~~~~~~~~~~~~~~~~~~

1. Set ``config_name`` in ``examples/sft/config/custom_sft_openpi.yaml``:

   .. code:: yaml

      actor:
        model:
          openpi:
            task: sft
            config_name: "pi0_custom"

2. Register ``pi0_custom`` in
   ``rlinf/models/embodiment/openpi/dataconfig/__init__.py`` (a template
   is already there):

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

3. Adapt ``CustomDataConfig`` in
   ``rlinf/models/embodiment/openpi/dataconfig/franka_dataconfig.py`` to
   your keys and action space.

Normalization statistics
~~~~~~~~~~~~~~~~~~~~~~~~

For a newly collected LeRobot dataset, compute ``norm_stats`` for ``state``
and ``actions`` before SFT (especially important for real-robot data):

.. code:: bash

   # Local dataset directory (contains meta/info.json):
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id /path/to/realworld_franka_bin_relocation

   # Or a Hugging Face repo id (cached under ~/.cache/huggingface/lerobot):
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

- ``--repo-id`` is a local path or a LeRobot HF repo id; ``HF_LEROBOT_HOME``
  changes the cache parent.
- ``--config-name`` must match the training dataconfig.
- ``calculate_norm_stats.py`` writes to
  ``./assets/<config_name>/<repo_id>/norm_stats.json``
  (``TrainConfig.assets_dirs / repo_id``). Training and eval do **not** look
  next to SFT ``full_weights.pt`` for that file. Set
  ``openpi_data.norm_stats_path`` to that ``norm_stats.json``. If the key is
  omitted, OpenPI loads ``{assets_dir}/{asset_id}/norm_stats.json`` from the
  TrainConfig (after ``model_path`` is applied) and logs a warning. The
  RoboTwin recipe pins stats beside the base weights at
  ``physical-intelligence/robotwin/<task>/norm_stats.json``.

If standard deviations are tiny or the q99–q01 range is very narrow, widening
them often stabilizes training, especially when SFT is followed by online RL.

BEHAVIOR does not use that script. Point
``actor.model.openpi_data.norm_stats_path`` at an existing
``{assets}/{asset_id}/norm_stats.json``.

Installation
------------

.. include:: _setup_common.rst

**Option 1: Docker** — image ``agentic-rlinf0.4-maniskill_libero``:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # Mainland China mirror: infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

   source switch_env openpi

**Option 2: Custom environment**:

.. code:: bash

   # Add --use-mirror in mainland China.
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

``--model openpi`` installs the official OpenPI **runtime** (transforms and
data pipeline). The trained network is still ``openpi``.

LIBERO and real-world Franka
----------------------------

Edit data paths and ``actor.model.model_path`` in the experiment YAML.
Typical overrides:

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
       num_action_chunks: 4          # env-executed length, not network horizon
       is_lora: False                # set True and lora_rank for LoRA
       openpi:
         task: sft
         embed_dtype: bf16
         config_name: "pi0_libero"   # use pi0_realworld for Franka

Launch:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openpi
   bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

Logs and checkpoints go under ``runner.logger.log_path``, saved every
``runner.save_interval`` steps at ``.../checkpoints/global_step_<N>/``.

Pi0.5 Base Checkpoint Download and Conversion
---------------------------------------------

The π₀.₅ JAX base checkpoint is available at
``gs://openpi-assets/checkpoints/pi05_base``. The downloaded directory contains
``params/`` and must be converted to the RLinf OpenPI layout before use as
``actor.model.model_path``. Replace the YAML's ``/path/to/...`` placeholders
with actual paths on your server.

Run from the RLinf repository root in the OpenPI environment with JAX / Orbax
installed. Downloading requires the Google Cloud CLI:

.. code:: bash

   mkdir -p /path/to/checkpoints/jax
   gcloud storage cp --recursive \
     gs://openpi-assets/checkpoints/pi05_base /path/to/checkpoints/jax/

Prepare task-specific ``norm_stats.json`` first (see normalization statistics
above). This example uses BEHAVIOR's 32-step horizon. ``--input-model`` must
point to the directory containing ``params/``:

.. code:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/checkpoints/jax/pi05_base \
     --input-norm-stats /path/to/task/norm_stats.json \
     --output-model /path/to/pi05_base_openpi \
     --output-norm-stats /path/to/pi05_base_openpi/task/norm_stats.json \
     --action-dim 32 \
     --action-horizon 32 \
     --max-token-len 200

The converter defaults to π₀.₅; do not add ``--no-pi05``. Other π₀.₅ SFT
recipes use the same workflow with ``--action-horizon`` matching their existing
configs: 10 for ManiSkill RLT, and 20 for Franka RLT and dual-Franka. Conversion
does not require changing action dimensions, image counts, or training settings
in the YAML.

Verify that the output directory contains ``model.safetensors`` and
``config.json``. Replace the ``actor.model.model_path`` placeholder with that
output directory and point ``actor.model.openpi_data.norm_stats_path`` at the
statistics for the same task, for example:

.. code:: yaml

   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         norm_stats_path: /path/to/pi05_base_openpi/task/norm_stats.json

Checkpoints already in the RLinf OpenPI layout need no further conversion.
Official OpenPI PyTorch checkpoints (with ``paligemma_with_expert.*`` keys)
still require ``openpi_pytorch_to_openpi`` even when they contain
``model.safetensors``. See ``rlinf/utils/ckpt_convertor/openpi/README.md`` for
that command.

Pi0.5 + BEHAVIOR-1K
-------------------

Filesystem paths live in the experiment config; the model shape comes from
``model/pi0_5``:

.. code:: yaml

   defaults:
     - model/pi0_5@actor.model
     - hybrid_engines/fsdp@actor.fsdp_config

The streaming loader reads only ``data:`` (no hidden defaults). Change at least:

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
       model_path: /path/to/pi05_base_openpi   # new-format fp32 base weights
       openpi:
         task: sft
         embed_dtype: bf16
       openpi_data:
         norm_stats_path: /path/to/assets/behavior-1k/2025-challenge-demos/norm_stats.json

- ``use_skill: false`` trains on the main-task text; ``true`` uses per-frame
  skill text from ``task_subtasks`` (collapsed orchestrators in the dataset
  are not valid labels).
- ``fine_grained_level`` / ``tolerance_s`` control streaming time alignment.

.. code:: bash

   bash examples/sft/run_vla_sft.sh behavior_sft_openpi_pi05

Pi0 + RoboTwin
--------------

Uses the official OpenPI / LeRobot map-style loader with
``config_name: pi0_aloha_robotwin``. Example data:
`adjust_bottle <https://huggingface.co/datasets/RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data>`_;
matching HF SFT weights:
`RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle>`_.

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

14-D ALOHA actions and three cameras; OpenPI pads actions to 32-D before the
model. ``openpi_data.norm_stats_path`` pins SFT and eval to the same
``norm_stats.json``. Change the task name in that path when switching tasks.

.. code:: bash

   bash examples/sft/run_vla_sft.sh robotwin_adjust_bottle_sft_openpi


Pi0.5 + SpiritAI
--------------------

Train all Pi0.5 parameters on two GPUs with
``spiritai_sft_openpi_pi05_full.yaml``. The recipe uses RLinf's FSDP and
flow-matching loss, with W&B logging through your existing login.
Select the dataset and its normalization statistics for each task.
If the global W&B CLI and this venv use different default service URLs, set
``WANDB_BASE_URL`` to the server used for your existing login before launch.
Credentials are matched by server hostname.

The Cartesian layout has 25 values: left pose (6), psi (1), gripper (1),
right pose (6), psi (1), gripper (1), torso pose (6), and base velocity (3).
The three cameras are ``cam_high``, ``cam_left_wrist``, and
``cam_right_wrist``. Current state is normalized and tokenized with the
prompt; future actions are padded to 32 dimensions with a horizon of 50.
The 256-token budget includes both prompt and state. Check longer task
instructions for truncation.

Use the task's own stats file. For a stationary FoldBox task, set
``zero_base_action_targets=true`` so base targets are zero before normalization:

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

This starts full-parameter training with global batch 32, 90,000 optimizer
steps, FP32 parameter updates, BF16 computation, gradient checkpointing, and
``full_shard``. ``JAX_PLATFORMS=cpu`` confines data utilities to CPU; PyTorch
still trains on GPUs. Inspect the W&B run's loss, gradient norm, learning rate,
and step time before leaving the run unattended.

Increase the per-GPU ``actor.micro_batch_size`` after measuring peak memory.
Change ``actor.global_batch_size`` independently to compare 32, 64, and 128.
At equal sample exposure, these correspond to 90,000, 45,000, and 22,500
optimizer steps. Set ``runner.max_steps``,
``actor.optim.total_training_steps``, warmup, and save interval explicitly
for each experiment.

For a measured two-GPU workflow that distinguishes accumulation from GPU
batching and checks the actual NCCL path, see
:doc:`Tune OpenPI SFT on PCIe GPUs <../../guides/openpi_pcie_sft>`.

The opt-in ``data.stateful_loader=true`` snapshots consumed data and worker
random states through the existing SFT checkpoint interface. It exposes finite
epochs so the SFT worker advances the distributed sampler each epoch.
Resume by adding ``+runner.resume_dir=/path/to/checkpoints/global_step_5000``
to the same training command. Keep data contents, transforms, stats, rank
layout, micro batch, worker count, and seed unchanged. Keep the dataset and
stats alongside the experiment record; they are not bundled in the weights.

On glibc systems, this recipe also sets
``data.worker_malloc_trim_interval=64`` to return eligible free heap pages in
each data subprocess. Zero disables it. This allocator setting does not alter
the data resume signature; see the sustained-memory checks in the
:doc:`PCIe tuning guide <../../guides/openpi_pcie_sft>`.

Deploy the resulting checkpoint through the existing SpiritAI container:
:doc:`Real-world evaluation <../../evaluations/guides/realworld>`.
RLinf RTC guides sampling at inference time, while this recipe trains the
ordinary flow-matching objective. It does not train on clean action prefixes
or use a postfix-only loss.

Train an Explicit Subtask Policy
--------------------------------

For SpiritAI box folding, use ``spiritai_sft_openpi_pi05_subtask_lora`` to
fine-tune an existing Pi0.5 base with two supervised views. This downstream
recipe follows the explicit high-/low-level factorization in the
`Pi0.5 paper <https://arxiv.org/abs/2504.16054>`_; its training objectives are
subtask text CE and continuous-action flow matching.

.. list-table:: Views Constructed During Loading
   :header-rows: 1

   * - Branch
     - Inputs
     - Target
   * - High level
     - Current images, normalized state, overall instruction
     - Current subtask text and EOS
   * - Low level
     - Current images, normalized state, current subtask
     - A 50-step action window, padded to 32 dimensions

Keep the LeRobot v2.1 layout. ``task_index`` resolves through
``meta/tasks.jsonl`` to the overall instruction; each parquet row also needs
a nonempty ``subtask`` string declared in ``meta/info.json.features``. The
loader preserves frame alignment and constructs the two views online. The
low-level text contains the subtask, while the high-level response is causal
and supervises only the subtask tokens and EOS. Overlength high-level
sequences raise an error instead of truncating labels.

Native LoRA adapts attention and FFN weights in both Gemma experts. The
recipe uses VLM rank/alpha 16/16 and Action Expert rank/alpha 32/32, matching
the expert-specific defaults in `OpenPI Gemma
<https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py>`_.
RLinf's :doc:`LoRA guide <../../guides/lora>` gives a general rank-32 example;
it does not specify a dual-expert Pi0.5 joint-SFT recipe. Vision,
the tied token embedding/output weight, action/time projections, norms, and
other base parameters stay frozen. Flow gradients reach the VLM adapters
through the action expert's attention. The tokenizer uses the existing
PaliGemma vocabulary; its high-level prefix ends in ``Subtask:``.

``ce_chunk_size`` limits the number of response tokens projected at once.
The projection and CE chunks execute inside one embedding/output-module
forward, so ranks with different response lengths enter that FSDP wrapper
the same number of times. Keep chunking inside this boundary when extending
the loss; rank-local chunk counts must not control sharded-module calls.
For the unequal-length failure, its fix, and the validation limits, see
:ref:`openpi-subtask-collective-order`.

This stationary FoldBox recipe sets ``zero_base_action_targets=true``,
matching the existing full-parameter FoldBox run and its normalization file.

.. warning::

   Keep preprocessing consistent with the statistics. A zero-width base
   action quantile range can magnify nonzero raw base commands. The
   hierarchical loader rejects zero-base statistics without the matching
   ``zero_base_action_targets=true`` transform.

Action windows retain the existing sampler and may cross a subtask boundary;
the current frame's label conditions the whole window.

The execution settings use FSDP1 ``full_shard`` with
``backward_prefetch=pre`` and ``gradient_checkpointing=false``.
``openpi.torch_compile=true`` compiles the existing GELU-GLU kernel;
it does not compile the whole model. Keep the recipe's FP32 parameter updates,
BF16 backbone computation, and FP32 reduction policies when comparing runs.

Start with a 100-update two-GPU check using the task's normalization statistics.
Use the existing venv's Python; the two runner overrides stop the check at 100
updates and disable checkpoint writes, including the final update:

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

This uses two FSDP ranks, micro batch 8 per GPU, global batch 64, and four
accumulation rounds. Every optimizer update contains 32 high-level records
and 32 low-level action windows. Both ranks follow the same alternating
branch order. Each loss is averaged over its own records; the configured
objective is ``text_loss_weight * CE + flow_loss_weight * FM``. Initial
weights are 1 and 10. These are starting hyperparameters, not evidence of
better box-folding performance. Track ``subtask_ce``, ``flow_loss``,
``hl_records_per_update``, ``ll_action_chunks_per_update``,
``hl_tokens_per_update``, and ``peak_allocated_gib``. Compare action-window
exposure as well as optimizer steps against flow-only training.

``time/training`` measures a complete optimizer update, including all four
accumulation rounds, data waiting, computation, and training communication.
It does not measure a single microbatch. The earlier full-parameter 64/8
recipe used four rounds and 64 action windows per update; this recipe uses
two HL and two LL rounds, with 32 action windows. Its text branch adds a
257,152-class vocabulary projection and CE. Frozen base weights still
participate in computation and FSDP parameter gathering. FSDP1 units that mix
frozen weights and adapters can also allocate and reduce gradients for the
whole flat parameter. See :ref:`openpi-joint-update-timing` for the earlier
32/2 run's measured timing and its limits; those timings do not validate 64/8.

The recipe keeps a 45,000-update scheduler and 750-update warmup during the
check. All 100 updates therefore remain in warmup. A completed check verifies
the training path and finite metrics, rather than convergence or post-warmup
stability. To start formal training, omit ``runner.max_steps=100`` and
``runner.save_interval=-1`` and select a fresh log path and experiment name.
The recipe then runs 45,000 updates and saves every 5,000 updates. Its AdamW
learning rate and minimum learning rate are both ``1e-5``, so the
``openpi_cosine`` schedule stays constant after warmup.

Keep ``stateful_loader=true`` and an even accumulation count of at least
two. Add ``+runner.resume_dir=/path/to/checkpoints/global_step_5000`` to restore
the saved optimizer, scheduler, loader branch position, and RNG states.
Resume requires matching data content, preprocessing, normalization, rank
layout, micro batch, workers, seed, and hierarchical settings.

After training, merge the native adapters before deployment:

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.merge_lora \
     --ckpt /path/to/checkpoints/global_step_5000 \
     --output /path/to/merged_pi05_subtask_lora

The export contains bare ``full_weights.pt``, ``norm_stats.json``, and
``openpi_training.json`` with the training settings and text template
version. Evaluate with ``realworld_spiritai_eval_pi05_subtask_RTC`` and point
its model and normalization paths at that directory. Keep the training
token budget and state convention, and use ``openpi.task=eval``. Loading
unmerged adapter weights as a bare policy raises an error.

Inference generates a subtask first, then builds a fresh low-level context
from the current observation and that text. The high-level KV cache is
separate from the action cache. Direct policy calls regenerate each time;
callers that retain ``SubtaskPlanningState`` must reset it at episode
boundaries. The RTC worker owns this state and clears it on ``bootstrap``;
``planning_interval=3`` refreshes the planner every three action requests.
A changed subtask clears guidance from the previous action chunk. Generated
text and planning time are logged; an empty subtask stops action generation.
Keep real-robot success and generalization across box models as separate
evaluations after validating the training and inference path.

Evaluation
----------

Flow-only SFT writes ``full_weights.pt`` under
``.../checkpoints/global_step_<N>/actor/model_state_dict/``. ``openpi``
eval can load that ``.pt`` (or the ``global_step_<N>`` directory) directly:
wrapper / FSDP prefixes are stripped in memory, and the eval YAML supplies the
model shape. You do **not** need to convert to ``model.safetensors`` first.

Do not copy training ``openpi.task: sft`` into eval. Eval configs must use
``task: eval`` (that selects ``Pi0Eval`` and observation transforms). Keep
``config_name`` and ``num_action_chunks`` aligned with training.

Norm stats also do **not** travel with ``full_weights.pt``. Stock eval YAMLs
often interpolate ``openpi_data.norm_stats_path`` as
``${rollout.model.model_path}/.../norm_stats.json``, which assumes a converted
directory that already bundles stats. If you only retarget ``model_path`` at
an SFT checkpoint, also point stats back at the training copy (RoboTwin: next
to the base weights at ``physical-intelligence/robotwin/<task>/norm_stats.json``;
BEHAVIOR: the training ``openpi_data.norm_stats_path``).

.. code:: bash

   # Example: evaluate the SFT .pt in place (keep training-time norm stats)
   bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
     rollout.model.model_path=/path/to/logs/.../checkpoints/global_step_30000 \
     rollout.model.openpi_data.norm_stats_path=/path/to/pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

Eval steps: :doc:`BEHAVIOR-1K <../../evaluations/guides/behavior>` and
:doc:`RoboTwin <../../evaluations/guides/robotwin>`. To export a shareable
bare ``Pi0`` directory (``model.safetensors`` + ``config.json`` + copied
``norm_stats.json``), use ``sft_to_openpi``; commands are in
``rlinf/utils/ckpt_convertor/openpi/README.md``.

Visualization
-------------

Watch the **training loss**. Metric definitions:
:doc:`Training metrics <../../reference/metrics>`.

.. code:: bash

   tensorboard --logdir ./logs
