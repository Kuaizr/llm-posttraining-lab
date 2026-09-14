# 实验报告：C00-QWEN-LORA-STEP-001

## 1. 问题与假设

- 所属章节：C00。
- 本次只改变什么：在已验证 assistant mask 的单条工具轨迹上，对真实 Qwen3.5-0.8B 的语言模块 LoRA 执行一次 optimizer update。
- 对照组与父 checkpoint：`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`；无效果对照组。
- 假设与可能的反证：若 loss/梯度有限、存在非零 LoRA 梯度、optimizer 后参数变化且 adapter 可保存重载，则真实模型最小训练链路可用；OOM、NaN/Inf、全 mask、零梯度、参数不变或重载不一致均反证。
- 不变的评测任务/预算：1 条虚构样本、97 token、50 个有效 label、恰好 1 次 backward 和 1 次 optimizer step。

## 2. 数据与反馈契约

- 数据/任务生成器版本、随机种子、hash：`scripts/inspect_tokens.py` 中固定 5-message 轨迹；canonical JSON SHA-256 `bbc1e285538245a1dc27a809b64357410f2a8766b8de2b9c4dd6fb51cfe62717`；训练 seed `20260915`。
- train / dev / test-IID / test-OOD 数量：train smoke=1；其余 `NOT_RUN`。
- family/instance 隔离方式与去重检查：不适用；单条手写虚构样本，不作为效果数据集。
- 本次是否接触封存 test：否；尚未建立 test。
- 教师模型/版本/提示词版本/来源：无教师模型；无 API 调用。
- 实际工具输出的保存位置与重放测试：工具返回为固定虚构 observation；未调用真实工具。
- 数据过滤数量与拒绝原因：0；脚本在长度超过 128 时拒绝截断。
- verifier/reward 版本、边界测试与反例：不适用；SFT causal LM loss。
- gold/oracle 是否可能进入模型输入：tool observation 作为条件输入存在，但 label 为 `-100`；assistant 工具调用和最终回答是训练目标。

## 3. 模型、环境与配置

| 项目 | 实际配置 |
|---|---|
| OS / GPU / 驱动 / Python | Arch Linux on WSL2；RTX 5070 Ti 16303MiB；KMD 610.47；Python 3.12.13 |
| PyTorch / CUDA / Transformers / TRL / PEFT | torch 2.14.0+cu130；CUDA runtime 13.0；Transformers 5.17.0；TRL 1.13.0；PEFT 0.20.0 |
| 模型 ID / revision / 父 checkpoint | `Qwen/Qwen3.5-0.8B` / `2fc06364715b967f1860aea9cf38778875588b17` / 同一固定本地 snapshot |
| tokenizer / 模板版本 / thinking 设置 | snapshot 自带 processor；TRL Qwen3.5 nothink training template；`enable_thinking=False` |
| LoRA / 量化 / dtype | r=4、alpha=8、dropout=0、bias=none；12 类语言投影后缀；无量化；BF16 |
| 序列长度 / 生成长度 / micro-batch / 梯度累积 | 97 / 不生成 / 1 / 1；max length 128；packing 关闭（手工单样本） |
| 优化器 / 学习率 / seed / optimizer steps | PyTorch AdamW，默认 weight decay 0.01；lr `1e-4`；seed `20260915`；1 step |
| GRPO group size / 采样配置 / old/ref 来源 | 不适用 |
| 最大工具调用 / 回合数 / 超时 | 固定轨迹 1 次工具调用；外层硬超时 10 分钟 |
| 训练停止条件与资源授权 | 2026-09-15 获学习者授权；OOM、非有限 loss/梯度、全 mask、零梯度、参数不变或重载校验失败即停止 |

attention implementation 为 SDPA。模型报告 `causal_conv1d` 和 `flash-linear-attention` 未安装，实际使用正确但较慢的 PyTorch fallback。

LoRA target modules：

```text
in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj,
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

## 4. 实际执行

```text
source scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
timeout --signal=TERM 10m python -u scripts/smoke_qwen_lora_step.py
```

- 代码快照/hash：`scripts/smoke_qwen_lora_step.py` SHA-256 `a65720eb3c5a33ac3a5957d3f1d525d51c0fc4a13f08ce062a3df3196021d47d`；执行时尚未提交。
- 配置文件路径：配置常量固定在 `scripts/smoke_qwen_lora_step.py`。
- 日志/轨迹/checkpoint 路径：终端输出；adapter 为 `runs/C00-QWEN-LORA-STEP-001/adapter/`（Git 忽略）。
- 保存→重载验证：PEFT 保存并在新 base 实例上重载；逻辑 adapter checksum `36a47e113de40dabb35919c567873d84fd70823027a984623c6d0727237ab523` 一致。
- adapter 文件：`adapter_model.safetensors` 10,871,936 bytes；文件 SHA-256 `72885aa7a51b247fa3637ec7f2367cd06408764b2ebe3874bf949ea1dfc688d4`。
- 实际执行测试与输出：模型类 `Qwen3_5ForCausalLM`；loss `0.9216660261154175`；372 个可训练张量、2,705,664 个参数；186 个张量有非零梯度；step 后 372 个张量变化；保存重载 PASS。
- 未运行项及原因：第二步训练、生成评测、收敛检查、效果对照均未授权且不属于 smoke。
- 执行程度：`SMOKE_ONLY`。

`changed_trainable_tensors=372` 不能全部解释为非零训练信号：只有 186 个梯度张量非零，AdamW 的 decoupled weight decay 也可能改变梯度为零但参与 optimizer 的参数。

## 5. 资源与成本

- 实测峰值显存与测量方式：`torch.cuda.max_memory_allocated=2387.39MiB`；`max_memory_reserved=2442.00MiB`。
- 实测运行时长/吞吐：脚本 `main()` 内 14.59 秒；包含 WSL/D 盘 Python 冷启动的外部墙钟约 115 秒；单样本 smoke，不报告训练吞吐。
- 教师请求数、输入/输出 tokens、重试和缓存命中：0 请求；无外部生成。
- 已知实际 API 费用：0；未调用 API。
- 价格未知部分：不适用。
- 是否达到预算/停止上限：未达到；1 step 后按授权边界正常停止，远低于 10 分钟硬超时。

## 6. 真实评测结果

| 指标 | 对照 | 实验 | 分母/样本数及定义 |
|---|---|---|---|
| 任务成功率 | NOT_EVALUATED | NOT_EVALUATED | 无独立评测 |
| 无效工具调用率 | NOT_EVALUATED | NOT_EVALUATED | 无生成 |
| 错误恢复成功率 | NOT_EVALUATED | NOT_EVALUATED | 无生成 |
| 平均工具调用数 | NOT_EVALUATED | NOT_EVALUATED | 无生成 |
| 平均生成 tokens / 截断率 | NOT_EVALUATED | NOT_EVALUATED | 无生成；训练样本未截断 |
| 独立 OOD 成功率 | NOT_EVALUATED | NOT_EVALUATED | 尚未建立封存评测 |

- 本次评测用 dev 还是 test：均未使用。
- 重复运行/随机种子：只运行一次；seed `20260915`。
- 不确定性、样本量限制：样本数 1、optimizer step 1，只能验证链路。
- 训练 reward 与外部成功率是否一致：不适用。
- 不能据此得出的结论：不能证明 loss 会持续下降、模型学会工具调用、任务成功率提升、配置可长期稳定训练，或 2B/DPO/GRPO/PPO 能在 16GB 上运行。

## 7. 失败分析

无执行失败。较慢 kernel fallback 是性能事实，不是正确性失败；当前 smoke 无需安装优化扩展。

## 8. 结论

- 效果证据：`NOT_EVALUATED`。
- 学到了什么：固定 Qwen3.5-0.8B、本地 assistant-only labels、语言层 LoRA、BF16 和当前 CUDA 栈能够完成真实 forward/backward/单次 optimizer update 及 adapter 保存重载。
- 当前主要瓶颈与证据：训练正确性与效果尚未评估；只有 1 条样本和 1 次更新。
- 下一次只改变什么：不再运行训练；由学习者独立解释单步 smoke 的证据边界。
- 是否需要新的授权/算力：当前下一步不需要；任何新增 GPU step 仍需重新授权。
