# 训练配置与验证记录

用本记录恢复选定的双卡配置、辨认历史测试，并手动启动、恢复或导出本轮实验。下面命令使用现有 `.venv` 与 RLinf 原生入口；本次工程备份只核对命令，没有执行训练或真机动作。

## 最终参数从哪里来

先前核对了全参数实际运行 `logs/20261028_095838-spiritai_foldbox_pi05_full_bs64`，确认 global batch 为 64、每卡 micro batch 为 8。通用 YAML 的默认 32/1 没有反映该运行的 CLI 覆盖。因此本轮以真实配置对齐 batch、优化器与总预算，再保留明确的 LoRA 和联合目标差异。全参数与 LoRA 数据重复 episode 的差异按用户要求暂不展开。

当前配置为 [`spiritai_sft_openpi_pi05_subtask_lora.yaml`](../../../examples/sft/config/spiritai_sft_openpi_pi05_subtask_lora.yaml)，继承 [`spiritai_sft_openpi_pi05_full.yaml`](../../../examples/sft/config/spiritai_sft_openpi_pi05_full.yaml)。

| 参数 | 最终值 / 说明 |
|---|---|
| 硬件 / placement | 本机 2 × RTX PRO 5000 Blackwell，单卡 48935 MiB；actor `0-1` |
| Global / micro / accumulation | 64 / 每 rank 8 / 每 rank 4 轮 |
| 每次更新目标数 | 32 HL records + 32 LL action windows |
| 总训练 / scheduler horizon | 45000 / 45000 optimizer updates |
| Warmup / 保存 | 750 / 每 5000 步；5000、10000、…、45000 共 9 个计划保存点 |
| AdamW | lr=min_lr=`1e-5`，betas=`0.9/0.95`，eps=`1e-8`，weight decay=`1e-10`，clip=`1.0` |
| Scheduler | `openpi_cosine`；因 lr=min_lr，warmup 后为常数 |
| LoRA rank/alpha | VLM 16/16，Action Expert 32/32；attention + FFN |
| CE/FM | 权重 1/10，分支 1:1 交替，CE chunk 32 |
| 输入与动作 | 三图像 224×224，25D Cartesian，模型 pad 32D，horizon 50 |
| 文本与 state | max token 256，subtask 上限 96，discrete state 开启 |
| 数据 | 每 rank 4 workers，stateful loader，shuffle/drop-last，seed 42，heap trim 64 |
| 参数与精度 | FP32 参数更新/optimizer/reduction/buffer，BF16 主干计算，保留 FP32 projection/norm/vision embedding 规则；AMP/scaler 关闭 |
| FSDP | FSDP1 `full_shard`，`use_orig_params=true`，`limit_all_gathers=true`，CPU offload 关闭 |
| 最终执行开关 | `gradient_checkpointing=false`、`backward_prefetch=pre`、`openpi.torch_compile=true` |
| 编译范围 | 原有 GELU-GLU 局部算子，整个模型没有被 compile |
| 运行环境 | `NCCL_P2P_LEVEL=PHB`、`OMP_NUM_THREADS=4`、`JAX_PLATFORMS=cpu`；W&B 使用已有服务地址 |

LoRA rank 取现有 OpenPI 双 expert 的推荐值；没有把 RLinf 通用 rank-32 示例解释成两个 expert 都必须用 32。本轮没有改变 native LoRA 初始化、precision policy 或模型 attention 实现来换取吞吐。

## 调优做过哪些尝试

有界测试保持 batch、数据、loss、LoRA 与优化器一致，通过已有 FSDP 和模型开关比较执行方式。探索已按用户要求停止，选用最快的已完成候选；这不是全配置空间最优性的声明。

| 配置 | 已完成更新 | 计时窗口 | 平均秒 / 完整更新 | 结论 |
|---|---:|---|---:|---|
| 对齐基线：full_shard、重计算开、无编译/预取 | 100 | 51–100 | 6.510 | 最终对照 |
| full_shard、关闭重计算 | 20 | 6–20 | 5.312 | 节省重计算，增加 activation 容量 |
| shard_grad_op、重计算开 | 20 | 6–20 | 5.785 | 有收益但不及选定配置 |
| no_shard、重计算开 | 20 | 6–20 | 5.954 | 未选；该路径还会选择 fused AdamW，不能作为纯通信隔离对照 |
| full_shard、关闭重计算、局部编译 | 20 | 6–20 | 5.070 | 保留局部编译 |
| shard_grad_op、关闭重计算、局部编译 | 0 | 首次更新 | OOM | 淘汰 |
| 上述 full_shard 编译候选 + expandable_segments | 20 | 6–20 | 5.084 | 无额外速度收益，未带入正式环境 |
| full_shard、关闭重计算、局部编译、backward prefetch | 20 | 7–20 | 4.780 | 第 6 步采集 profile，计时排除该步 |
| 最终选定组合，无 profiler | 100 | 51–100 | 4.831 | 正式 recipe 的有限步数依据 |

20 步表格的 6–20 基线均值为 6.420 秒，7–20 基线为 6.432 秒。不要拿不同窗口的小数差值计算单个开关收益；正式结果使用两次 100 步的同一 51–100 窗口。其他 profiling 预检没有成功生成 trace，额外 vision grouping 运行不参与最终选择。[完整小型摘要](artifacts/execution_tuning_summary.json)

最终结果为 6.510 → 4.831 秒/更新，耗时减少 25.78%，吞吐为 1.347 倍；records/s 为 13.247，LL action windows/s 为 6.623。按此短测均值外推，45000 步约 60.4 小时纯训练时间，未包含保存、中断或长程波动，不能作为完成时间承诺。

一次双 rank GPU profile 的计算 kernel 时间并集约为 4.04–4.07 秒，NCCL 与计算重叠后独占约为 0.46–0.58 秒。它支持“通信与计算均有成本”的判断；不能相加 CPU range、operator 总量和 GPU kernel 时间，也不能据单个 profile 给 CE 单独分配端到端耗时比例。

## 100 步验证证明了什么

最终运行目录为 `logs/pi05_exec_best_100_20261102_231803`，W&B run ID 为 `cqmk887u`；对齐基线为 `logs/pi05_exp_c_bs64_m8_100_20261102_205203`，run ID 为 `94cij5p9`。二者除日志字段外只改变重计算、backward prefetch、局部编译三个配置。[最终验证](artifacts/selected_execution_100_step_validation.json)

最终 100/100 次更新完成，exit 0，无 timeout/OOM，loss、CE、FM、grad norm、LR 均为有限值。HL token 数和 LR 与基线一致；BF16 编译改变舍入，loss/梯度并非逐位一致。这项比较没有检验最终训练质量等价。

跨 rank PyTorch peak allocated 为 38.4266 GiB，NVML 采样峰值为 48229/48357 MiB。测试关闭了所有 checkpoint 写入，实际没有生成权重文件或 checkpoint 目录。全部 100 步仍在 warmup 内：未覆盖 warmup 后稳定性、当前执行组合的保存/恢复、learned subtask 生成质量或真机性能。

当时还通过了 6 个 focused CPU model tests、配置解析和中英文文档构建；更早的 collective 修复通过 166 个组件测试、2 个已说明 skip。记录不同测试的发生阶段，避免把所有验证拼成对当前配置的完整训练保证。

## 恢复本机资产并手动启动

正式运行依赖以下外部资产；Git 代码和工程 JSON 不能替代它们。保持数据、基础模型和 stats 的配对关系，使用[备份指纹](artifacts/engineering_backup_manifest.json)核对已记录文件。

| 资产 | 本机路径 |
|---|---|
| 数据 | `/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_10subtasks` |
| Base checkpoint | `/home/deng/Documents/codebase/RLinf/model_checkpoints/pi05_base_rlinf_pytorch` |
| Stats | `/home/deng/Documents/codebase/RLinf/model_checkpoints/foldbox_assets/norm_stats.json` |
| Python 环境 | `/home/deng/Documents/codebase/RLinf/.venv/bin/python` |

在 tmux 会话内手动执行下面命令即可启动新的 45000 步运行。此处保存此前交付的运行方式，未替用户执行；正式训练仍由用户控制。

```bash
cd /home/deng/Documents/codebase/RLinf

pi05_run_dir="$PWD/logs/$(date +%Y%m%d_%H%M%S)-pi05_subtask_lora_bs64"
mkdir -p "$pi05_run_dir/wandb"
set -o pipefail
git rev-parse HEAD > "$pi05_run_dir/git_commit.txt"
git status --short > "$pi05_run_dir/git_status.txt"

JAX_PLATFORMS=cpu \
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
NCCL_P2P_LEVEL=PHB \
OMP_NUM_THREADS=4 \
WANDB_BASE_URL=https://api.forge.coreweave.com \
WANDB_DIR="$pi05_run_dir/wandb" \
RAY_ADDRESS=local \
RAY_TMPDIR="$(mktemp -d /tmp/pi05formal.XXXXXX)" \
.venv/bin/python -u examples/sft/train_vla_sft.py \
  --config-name spiritai_sft_openpi_pi05_subtask_lora \
  data.train_data_paths=/home/deng/Documents/dataset/20260805_FoldBox_SpiritAI_Moz1WB_10subtasks \
  actor.model.model_path="$PWD/model_checkpoints/pi05_base_rlinf_pytorch" \
  actor.model.openpi_data.norm_stats_path="$PWD/model_checkpoints/foldbox_assets/norm_stats.json" \
  runner.max_steps=45000 \
  actor.optim.total_training_steps=45000 \
  actor.optim.lr_warmup_steps=750 \
  runner.save_interval=5000 \
  runner.logger.log_path="$pi05_run_dir" \
  runner.logger.experiment_name=spiritai_pi05_subtask_ce_flow_lora_bs64_45000 \
  2>&1 | tee "$pi05_run_dir/train.log"
```

命令使用独立 Ray 临时目录、指定 W&B endpoint，并同时保存终端输出和 Git 状态。`PHB` 是本机测过的拓扑配置，换机器后先核对 P2P 能力和实际 NCCL transport。checkpoint 位于 `$pi05_run_dir/spiritai_pi05_subtask_ce_flow_lora_bs64_45000/checkpoints/`；`runner.logger.log_path` 还会再附加 experiment name，不能只在 log root 下寻找权重。训练 stdout 会打印配置；需要独立解析快照时，在同样参数后加 `--cfg job --resolve` 并保存 stdout，该选项只解析配置。

若以后重做 100 步检查，使用新日志目录，将命令中的 `runner.max_steps=45000` 替换为 `100`、`runner.save_interval=5000` 替换为 `-1`。保持 `actor.optim.total_training_steps=45000` 和 warmup 750，避免测试与正式优化器 schedule 分叉。不要把这些测试覆盖带入正式命令。

## 从 Checkpoint 恢复与导出

训练恢复要求完整 checkpoint 和相同的 rank 布局、microbatch、worker 数、seed、数据、预处理、stats、LoRA 与 hierarchical 设置。选择已有保存点，在上述训练命令中增加：

```text
+runner.resume_dir=/absolute/path/to/checkpoints/global_step_5000
```

这里的行首一个 `+` 是 Hydra 新增配置字段的语法。总目标仍为 45000 步，不能把恢复后的 remaining steps 填成新的总目标。loader 会校验恢复契约；出现不匹配时先核对实际配置，不能为了继续运行而绕过检查。

部署时改用原生合并工具，输出到一个新的目录：

```bash
cd /home/deng/Documents/codebase/RLinf
.venv/bin/python -m rlinf.utils.ckpt_convertor.openpi.merge_lora \
  --ckpt /absolute/path/to/checkpoints/global_step_5000 \
  --output /absolute/path/to/merged_pi05_subtask_lora
```

输出包含 bare `full_weights.pt`、`norm_stats.json`、`openpi_training.json`。工具拒绝覆盖已有部署权重；合并目录用于推理，原训练 checkpoint 继续用于恢复。评估选择 [`realworld_spiritai_eval_pi05_subtask_RTC.yaml`](../../../evaluations/realworld/realworld_spiritai_eval_pi05_subtask_RTC.yaml)，将 model/stats 指向合并目录，保持 `openpi.task=eval` 和 `is_lora=false`。

下一次记录结果时，分别补充实际运行配置、warmup 后曲线、保存与新进程恢复、生成 subtask 的阶段正确性，以及不同型号盒子的真机成功率。当前归档没有这些结果，不能从工程通过推导它们。
