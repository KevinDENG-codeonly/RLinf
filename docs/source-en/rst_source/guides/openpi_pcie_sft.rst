Tune OpenPI SFT on PCIe GPUs
==============================

Use this guide to measure and tune full-parameter OpenPI SFT when PCIe
communication and memory capacity limit a multi-GPU run. Start from the
:doc:`OpenPI SFT recipe <../examples/embodied/sft_openpi>`, verify the actual
transport, increase the per-GPU micro batch, and validate checkpoint recovery
before committing a long training budget.

The measurements below come from a 2026-10-27 SpiritAI FoldBox experiment on
two RTX PRO 5000 Blackwell GPUs, each reporting 48,935 MiB, with about 125 GiB
host RAM. The model had 3,353,433,872 parameters. Its three-camera, 25-D
Cartesian data were padded to the network's 32-D action space, with horizon 50
and a 256-token prompt/state budget. These are capacity and throughput results
for one machine; convergence and real-robot success require separate experiments.

.. warning::

   A subsequent 5,000-update formal attempt stopped after update 506: eight
   DataLoader subprocesses accumulated about 97.58 GiB, and Ray terminated
   both actors at its 95% host-RAM threshold. No checkpoint had been saved
   before the configured interval of 2,500. The short benchmarks below do
   not establish long-run data-loading stability. Validate sustained real-data
   loading before restarting, and save recovery points early in the next run.

Start From a Comparable Baseline
----------------------------------

Compare runs at the same global batch before attributing a change in seconds
per step to an optimization. An optimizer update contains multiple forward and
backward passes when gradients are accumulated:

.. math::

   G = N_{GPU} \times M \times A

Here, ``G`` is the global batch, ``M`` is the per-GPU micro batch, and ``A`` is
the number of accumulation rounds. With two GPUs and global batch 64, micro
batch 1 requires 32 rounds per GPU; micro batch 8 requires four. Increasing
only the global batch increases accumulation, rather than the size of each
GPU computation.

Keep the model, observations, horizon, normalization, seed, optimizer and
precision policy fixed while measuring a performance change. This experiment
used FP32 parameter storage and optimizer updates, BF16 backbone computation,
FP32 reduction, the existing OpenPI FP32 module policies, activation
checkpointing, and FSDP1 ``full_shard``. Local kernel compilation was disabled
to focus on PCIe. The framework has FSDP2 support, but the OpenPI sharded path
in this version validates FSDP1; changing the backend is not a drop-in tuning
step. Bounded tests used constant learning rate ``1e-5`` and zero warmup to
observe updates; the formal plan uses 750 warmup updates. Throughput checks do
not validate that optimization schedule.

The shared SpiritAI YAML keeps conservative defaults of global batch 32 and
micro batch 1. Record the resolved runtime configuration, including CLI
overrides: selecting that YAML by name alone does not select the tuned
64/8 configuration.

Verify the Transport
----------------------

A GPU pair can support peer access while NCCL chooses a different transport.
Read the topology and peer-access capabilities, then inspect the channels used
by a bounded training run:

.. code-block:: bash

   nvidia-smi topo -m
   nvidia-smi topo -p2p r
   nvidia-smi topo -p2p w
   nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv

   mkdir -p logs
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,ENV
   export NCCL_DEBUG_FILE="$PWD/logs/nccl.%h.%p.log"

These commands read the hardware and enable per-process NCCL diagnostic files
for the next launch. In this case both links were Gen5 x16 and the topology
was ``NODE`` within one NUMA node. Peer reads and writes were supported, but
NCCL's GPU channels initially used ``SHM/direct``. The initialization message
``Using network Socket`` alone does not identify the intra-node payload path.

For this topology, allowing PHB peer access changed the channels to
``P2P/IPC``:

.. code-block:: bash

   export NCCL_P2P_LEVEL=PHB

PHB permits peer traffic through the host bridges within the NUMA node. It
does not create an NVLink connection. Keep this as a measured per-machine
launcher setting, and repeat the correctness and throughput checks on another
machine. NCCL normally selects a cutoff from the architecture and environment;
see the `NCCL transport settings
<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-level>`_.

Separate byte units and bandwidth definitions when interpreting the result.
26.7 GB/s is about 214 Gb/s. PCIe Gen5 x16 has a theoretical one-way rate of
about 63.0 GB/s before transaction overhead. The measured one-way Tensor copies
were 41.3–41.7 GB/s. Two-rank all-gather and reduce-scatter use
``busbw = algbw × (n-1)/n``, so their bus bandwidth is half the algorithm
bandwidth. These collectives are different workloads from a one-way copy.
See `PCI-SIG's bandwidth table
<https://pcisig.com/sites/default/files/files/PCI-SIG-Seamless_Transition_to_PCIe_5.0_in_System_Implementations_FINAL.pdf>`_
and the `NCCL Tests measurement definitions
<https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md>`_.

Reduce Repeated Parameter Transfers
-------------------------------------

After verifying P2P, increase the micro batch while keeping the global batch
fixed. This reduces repeated parameter all-gathers and launch overhead, and
changes matrix computation sizes. Measure the combined effect rather than
assigning every gain to PCIe bandwidth.

.. figure:: ../../../experiments/pi05-flow-only-full/artifacts/pcie_sft_tuning.svg
   :width: 100%
   :alt: Controlled step times and sampled GPU-memory headroom for OpenPI SFT.

   Each comparison has eight updates, with three warmup updates discarded.
   Ranges show the min–max of five timed updates, not confidence intervals.
   NVML occupancy includes desktop use and may miss shorter peaks.

.. list-table:: Fixed global batch 64
   :header-rows: 1

   * - Transport
     - Micro batch / GPU
     - Accumulation / GPU
     - Mean training seconds / update
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

Changing only transport at micro batch 1 reduced training time by about 12.9%.
The selected micro batch 8 configuration then averaged 6.695 seconds over 17
timed updates in a separate 20-update confirmation run, about 4.19 times the
initial baseline throughput. Native ``time/training`` includes data waiting,
forward/backward and optimizer work, while excluding checkpoint writes and
model initialization.

Select for the whole resource lifecycle. Micro batch 32 reached 48,151 MiB in
the short run, leaving only about 784 MiB of sampled headroom. Micro batch 8
was selected and tested through saving and restart; its restart run reached
48,130 MiB, and repeated saves reached 48,276 MiB. A steady training sample
does not describe the checkpoint or restart peak.

Other existing controls did not improve the chosen tradeoff on this machine:
backward prefetch left the baseline near 28 seconds; ``shard_grad_op`` at micro
batch 1 took 15.180 seconds and reached 48,327 MiB. Fixing NCCL at 16 CTA
improved a 256 MiB all-gather microbenchmark to about 29.7 GB/s, but changed
actual micro batch 8 training from 6.526 to 6.501 seconds while adding 1,163 MiB
to sampled peak occupancy. NCCL's default CTA choice was retained.

Within the same micro batch, the recorded loss and gradient norms matched
for the transport/prefetch/sharding comparisons. Changing micro batch changes
the grouping of random noise, times and augmentation draws, so cross-micro
loss sequences need not match. Short finite-loss checks do not establish
convergence. For kernel, communication and data-wait attribution, use
:doc:`Profiling <profile>`; these measurements used transport logs and
controlled runs, without a GPU timeline.

Prove Recovery and Memory Reuse
---------------------------------

A weights-only inference check does not establish resumability. Save the full
training state, exit the process, restore it, and inspect optimizer progress,
data position, RNG state and subsequent weight updates. See
:doc:`Resume Training <resume>` for the checkpoint layout.

The selected run saved step 20 and resumed to step 21. The audit found 665
active optimizer states advancing to 21; two structurally unused final prefix
norm states remained at zero. Consumed micro batches advanced from 80 to 84
per rank, and representative action-head, vision and language weights changed.
An uninterrupted-versus-resumed parameter-equivalence comparison was not
performed. Keep the micro batch, workers, seed, rank layout, data and stats
fixed when using the opt-in stateful loader's resume signature.

Check host RAM separately from VRAM. RLinf stages sharded training state on
CPU and exports full model weights on rank 0. One checkpoint occupied about
50 GiB, including about 12.5 GiB of FP32 full weights. Disk size is not the
same as additional live RAM, and GPU OOM is not solved by host swap.

.. figure:: ../../../experiments/pi05-flow-only-full/artifacts/checkpoint_host_memory.svg
   :width: 100%
   :alt: Host available RAM and anonymous memory during four checkpoint saves.

   The original 2 GiB swap was used for this test. System available RAM and
   process-cgroup anonymous RAM have different scopes and are not additive.

The seven-update test requested saving every two updates, and the final update
also saved: actual checkpoints were at 2, 4, 6 and 7. Across 578 samples at about
0.5-second intervals, available host RAM stayed above 52.53 GiB. Anonymous RAM
peaks were about 56.30, 57.19, 58.05 and 58.42 GiB, fell after saves, and host
available RAM returned to about 111 GiB after process exit. The roughly 2.1 GiB
increase is worth retaining in the record; this short run cannot prove zero
growth over days.

Deleting the four audited temporary checkpoints freed about 199.92 GiB of
disk, while available RAM stayed near 111.8–111.9 GiB and cache fell from about
65 to 15 GiB. File cache was already reclaimable. Distinguish it from live
process memory, named tmpfs objects and retained checkpoint files; use
``MemAvailable`` and the relevant process/cgroup counters rather than treating
all cache as a leak. The `Linux memory counters
<https://docs.kernel.org/filesystems/proc.html#meminfo>`_ and
`tmpfs documentation <https://docs.kernel.org/filesystems/tmpfs.html>`_
describe these different resources.

Swap was increased to about 32 GiB after the tests by retaining 2 GiB and adding
30 GiB. This later operating-system change is separate from the conditions that
passed the tests. The SFT runner retains checkpoint directories; account for
disk retention as well as RAM when diagnosing a failed weight write.

Bound Data-Worker Heap Retention
---------------------------------

Measure sustained data loading separately from checkpoint writes. In the
506-update failure, ordinary and stateful CPU loaders both accumulated memory;
Python garbage collection did not stop the growth. Returning free native-heap
pages to the operating system did. This identifies retained allocator pages
as a major contributor, without attributing every allocation to one library.

The SpiritAI recipe now enables per-process heap reclamation in the existing
OpenPI dataset wrapper:

.. code-block:: yaml

   data:
     num_workers: 4
     stateful_loader: true
     worker_malloc_trim_interval: 64

The interval counts samples read by each DataLoader subprocess, not optimizer
updates. Every 64 reads, the subprocess invokes glibc's
`malloc_trim <https://man7.org/linux/man-pages/man3/malloc_trim.3.html>`_
to return eligible free heap pages. It does not free live tensors, change
normalization, or draw random numbers. Calls occur only in data subprocesses;
with zero workers the option has no effect. The framework default is zero
(disabled), and enabling it requires the stateful OpenPI loader and an
allocator exporting glibc's function. An unsupported allocator reports an error
when reclamation is attempted.

The reclamation counter is outside the data checkpoint. Changing only this
interval leaves the sampler, worker RNG snapshots, and resume signature
unchanged. Keep the other resume constraints above. Verify the fix by consuming
at least the failed run's sample count per worker, monitoring RSS/USS and host
available RAM, then repeat bounded training and checkpoint recovery. Select
the checkpoint cadence for the experiment and budget its disk usage separately.

The fixed two-rank, eight-worker CPU test consumed 40,000 real samples, at
least 5,000 per worker, beyond the failed run's roughly 4,048. It completed
in 585 seconds with minimum available host RAM of 99.59 GiB and a sampled
maximum worker RSS of 2.70 GiB. This is data-read coverage equivalent to 625
global-batch-64 updates, not 625 model updates or proof of multi-day stability.

Keep the Budget and Runtime Evidence
--------------------------------------

Translate a recipe's sample budget when changing the global batch. The old
LoRA budget was ``32 × 90000 = 2880000`` windows. Equal exposure is 45,000
updates at global batch 64 or 22,500 at 128; warmup and save intervals should be
compared in the same units. That inherited budget is not a demonstrated
full-parameter convergence requirement.

The selected manual plan uses 45,000 scheduler updates, 750 warmup updates and
saving every 2,500 updates. A first stage can stop at 5,000 while retaining the
45,000-update schedule for resume. The confirmation timings imply about 9.3
hours of compute for that stage or 3.5 days for the full exposure budget,
excluding writes, interruptions and long-run variation. About 18 checkpoints
at 50 GiB each need about 900 GiB of disk.

This run targets sustained two-GPU engineering stability, including memory,
communication, data loading and checkpoint writes. Keep saving every 2,500
updates in both stages: the first 5,000 updates produce two checkpoints,
about 100 GiB, and the complete 45,000-update run produces about 18. Record
failures and their evidence as part of the experiment.

Keep these records together: the resolved Hydra configuration, complete source
version, normalization/data fingerprints, topology and NCCL path logs, timed
samples with warmup rules, and checkpoint audits. This experiment's HEAD also
had uncommitted changes, so a commit hash alone is insufficient. Archive small
results outside ignored log directories. Keep W&B credentials in the existing
login; if the CLI and venv SDK default to different service URLs, match
``WANDB_BASE_URL`` to the authenticated hostname.

Use tmux for terminal disconnects and checkpoints for process or machine loss.
Starting formal training and sending real-robot actions remain manual
operations. The SFT guide documents the data/model fields; the experiment
archive at ``docs/experiments/pi05-flow-only-full/README.md`` contains the exact
local launch and resume commands.

Download the Recorded Measurements
------------------------------------

Use the committed CSVs to rebuild tables after local logs are cleaned. The
original raw traces and model payloads are separate local artifacts. An AI
assistant helped run and document the finite tests.

* :download:`Training comparisons <../../../experiments/pi05-flow-only-full/artifacts/training_benchmarks.csv>`
* :download:`Communication microbenchmarks <../../../experiments/pi05-flow-only-full/artifacts/communication_benchmarks.csv>`
* :download:`Host-memory samples <../../../experiments/pi05-flow-only-full/artifacts/host_memory_samples.csv>`
* :download:`Checkpoint-memory audit <../../../experiments/pi05-flow-only-full/artifacts/checkpoint_memory.csv>`
* :download:`Validation summary <../../../experiments/pi05-flow-only-full/artifacts/validation_summary.json>`
* :download:`Fixed worker heap samples <../../../experiments/pi05-flow-only-full/artifacts/worker_heap_reclaim_soak.csv>`
* :download:`Heap reclamation and recovery audit <../../../experiments/pi05-flow-only-full/artifacts/worker_heap_reclaim_validation.json>`
* :download:`Revised manual configuration <../../../experiments/pi05-flow-only-full/artifacts/formal_resolved_heap_reclaim.yaml>`
