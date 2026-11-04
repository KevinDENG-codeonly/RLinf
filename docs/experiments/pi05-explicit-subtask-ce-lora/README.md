# 实验 C：显式 Subtask CE + Flow LoRA 工程备份

从这里恢复实验 C 的实现背景、数据契约、训练配置和故障处理方法。记录回答三个问题：这轮改了什么、为什么这样改、以后遇到同类问题应检查什么。

整理日期为 2026-11-04。实现与训练配置对应提交 [`3a7af6d1`](https://github.com/KevinDENG-codeonly/RLinf/commit/3a7af6d104afb24e0baf883ad1dd66c045040e69)，分支为 `feature/pi05-flow-explict-subtask-ce-lora`。性能与分布式验证来自 2026-11-02 的归档；本次补文档没有启动训练，也没有审计后续 45000 步运行的完成状态。

## 备份内容

按需要查阅以下记录；原始 JSON 的历史参数保留原样，以免把不同阶段的结果混在一起。

| 文档 | 可以恢复的信息 |
|---|---|
| [实现与数据契约](implementation.md) | 目标、Baseline 差异、annotation 分离、两条训练数据流、loss、LoRA 范围、推理与代码入口 |
| [问题与防复发记录](engineering-lessons.md) | 第 14 步卡死、W&B 重复认证、训练计时误读、显存口径和保存恢复注意事项 |
| [训练配置与验证记录](training-and-validation.md) | 最终参数、调优结果、100 步检查、45000 步手动命令、恢复和导出 |
| [证据目录](artifacts/README.md) | 数据转换、回归测试、计时、调优与源码指纹的来源和使用边界 |

本目录保存工程记录和小型证据。代码由上述 Git 提交保存；数据集、基础权重、训练 checkpoint 和完整日志仍是外部资产，恢复机器时需要一并保留它们。具体路径见[训练记录](training-and-validation.md)。`/tmp` 中的一次性转换和 profiling 脚本不是正式训练依赖。

## 这轮实际完成了什么

本轮在已训练的 π0.5 Base 上，为 box folding 增加显式的“先生成当前 subtask，再根据 subtask 生成动作”能力。目标泛化范围是不同型号的盒子。训练沿用 RLinf PyTorch、原生 LoRA、Ray SFT worker 和 FSDP，增加 Subtask CE 与 Flow Matching 的联合微调路径。

| 阶段 | 完成的工作 | 证据与边界 |
|---|---|---|
| 数据调整 | 将 overall 与逐帧 subtask 分离，保持 LeRobot v2.1 布局 | 712 episodes、1148747 frames；转换报告与官方 reader 抽样记录已归档 |
| 训练实现 | 两个 Gemma expert 的原生 LoRA、HL/LL 交替 microbatch、CE mask 与 flow loss | 代码、配置和相关单测随实现提交保存 |
| 推理实现 | 显式自回归 subtask、独立的动作条件、低频规划缓存和 RTC 状态处理 | 接口与测试存在；模型是否能生成合适的 subtask、真机效果仍需测量 |
| 分布式修复 | 将 CE chunk 循环移入一次 FSDP output-module 调用 | CPU/Gloo 2/4 rank 回归、旧实现负对照、人工确认越过原触发点 |
| 参数对齐 | 核实全参数实际日志后采用 global 64、micro 8、45000 步、warmup 750 | 参考的是实际运行配置，不能只看通用 YAML 默认值 |
| 执行调优 | 关闭重计算、启用 backward prefetch 和已有局部 GELU 编译 | 最终双卡 100 步通过，不保存权重；后 50 步平均 4.831 秒 |
| 正式启动准备 | 每 5000 步保存，提供现有 venv 可直接运行的命令 | 正式训练由用户手动启动 |

最终 100 步检查只证明该配置能完成有限次数更新、指标有限且无 OOM。100 步全部处于 750 步 warmup 内，不能据此宣称收敛、保存恢复通过或折盒成功率提高。

## 保留的研究边界

本轮检验的是 `π0.5 + LoRA + 显式分层联合微调` 这个方向。原有 LoRA baseline 在训练时输入 overall + 当前 subtask，推理时只输入 overall；实验 C 将当前 subtask 变成模型需要生成的中间结果。已有 [flow-only full 实验](../pi05-flow-only-full/README.md)提供训练配置和工程对照，但这些方向同时改变了训练目标、可训练参数或动作样本曝光量，不能直接分离单个因素的因果效果。

Base checkpoint 的既有规划能力、严格变量分离，以及额外 CE 是否改善动作能力，保留为待验证问题。Full Fine-Tuning、KI、FAST Action CE 和 Implicit HL 属于后续讨论范围；FineART 与 `Ke-Wang1017/openpi_subtask` 只提供工程参考。下次复用这份备份时，先恢复当前实现和配置，再决定是否扩展实验。

本记录由 AI assistant（Codex）根据对话、代码和本机归档协助整理；事实、推断和待验证项在各文档中分别标明。
