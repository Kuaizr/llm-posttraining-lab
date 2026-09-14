# C00 模型选择与 16GB 显存预算

> 日期：2026-09-14  
> 状态：三个模型已按固定 revision 下载并校验；0.8B Base forward、0.8B/2B 后训练版生成已实测。显存数字除明确标为“实测”外均为规划估算，不能替代 `torch.cuda.max_memory_allocated()`。

## 选择结论

本地先准备三个固定 revision：

| 用途 | 模型 | revision | 官方权重文件约 |
|---|---|---|---:|
| 从预训练基座做 SFT、比较 Base 与后训练起点 | `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 1.75 GB |
| Tool SFT、DPO、GRPO/PPO 最小闭环与 rollout | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 1.75 GB |
| SFT 成功后的规模迁移和生成评测 | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 4.55 GB |

暂缓 `Qwen/Qwen3.5-2B-Base`。当前阶段用 0.8B 的 Base/后训练对照即可回答 checkpoint 起点问题；再下载一个约 4.55GB 的 2B Base 不会扩大当前最小验收覆盖。

权重缓存固定在：

```text
/mnt/d/llm-posttraining-cache/huggingface/hub
```

不把权重提交 Git。

## 为什么 PPO 优先使用 0.8B

仅计算 BF16 参数权重，不含激活、KV cache、梯度和优化器状态：

```text
0.8B × 2 bytes ≈ 1.6 GB / 模型副本
2.0B × 2 bytes ≈ 4.0 GB / 模型副本
```

典型 PPO 可能同时涉及 policy、reference、reward、value，以及 policy 的训练状态。若四个组件都是同尺寸独立模型：

```text
0.8B：仅四份 BF16 权重约 6.4 GB
2.0B：仅四份 BF16 权重约 16 GB
```

第二行尚未计入激活、KV cache、CUDA workspace、梯度和优化器，因而 2B 多副本 PPO 不适合在这张 16GB 卡上作为顺滑的入门路径。即使是 0.8B，也应使用 LoRA、短序列、batch 1、顺序 rollout、可共享/卸载的 reference，以及程序 verifier 或更小 reward，之后以实际峰值显存决定能否扩大。

## 各训练方式的本地预期

| 方法 | 0.8B | 2B | 16GB 上的首选策略 |
|---|---|---|---|
| SFT LoRA | 预计宽松 | 短序列下预计可行 | 先 0.8B 跑通，再测 2B；batch 1、长度 256/512 起步 |
| 全参数 SFT | 加上 Adam 状态和激活后可能很紧 | 明显不适合 | 本课程不以全参训练为首个目标 |
| DPO LoRA | 预计可行 | 可能可行但取决于 reference 实现和序列长度 | 优先 PEFT 共享基座/禁用 adapter 作 reference，实测峰值 |
| GRPO | 无 value model，但有 policy/reference、rollout 和 KV cache | 需要激进降配或卸载 | 0.8B、短 completion、小 group，程序 verifier |
| PPO | 多模型副本和训练状态使预算最紧 | 本地 16GB 不作为目标 | 只用 0.8B 做极小闭环；必要时把 reward/reference 卸载到 CPU |

以上是容量规划，不是运行保证。Qwen3.5 是带视觉编码器的条件生成模型；即使任务仅使用文本，具体加载类是否保留视觉参数、LoRA target modules、attention 实现和训练器行为都要在实际 smoke 中确认。

## 降配顺序

发现 OOM 时按以下顺序处理：

1. 缩短 prompt、completion 和总序列长度。
2. batch 固定为 1，减少 rollout group 和并发。
3. 开 gradient checkpointing；关闭 cache 参与训练。
4. 使用 LoRA，仅训练必要语言模块。
5. reference/reward/value 顺序执行或 CPU offload，避免同驻 GPU。
6. 使用更小 reward 或程序 verifier。
7. 最后才评估 8/4-bit 量化训练；不预装 bitsandbytes。

## 下一次实测必须记录

- 完整模型 ID、revision、模型类和 tokenizer/chat template 版本。
- dtype、attention implementation、LoRA target modules。
- prompt/completion/max sequence length、batch、gradient accumulation。
- 每个阶段的 `torch.cuda.max_memory_allocated()` 与 `max_memory_reserved()`。
- 是否同时驻留 policy/reference/reward/value。
- 是否真实执行 forward、backward、optimizer step、保存和重载。

没有这些实测前，不宣称 2B PPO、GRPO 或任何配置“能在 16GB 稳定运行”。

## 0.8B 实际结构与 LoRA 兼容性

离线加载 `Qwen/Qwen3.5-0.8B@2fc0636…` 后统计：

```text
实际总参数：852,985,920
torch.nn.Linear 模块：237
```

模型同时包含 `model.visual` 与 `model.language_model`。因此纯文本工具任务不应未经检查直接使用 `target_modules="all-linear"`，否则视觉编码器线性层也会匹配。

使用下列语言模型投影后缀做了不含 backward 的兼容性 smoke：

```text
in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj,
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

结果：

```text
matched_modules=186
trainable_parameters=5,411,328
trainable_percent=0.6304%
adapter_size≈21.7MB
save_reload=PASS
```

这只证明该 target 集合能注入、保存和重载；最终是否同时训练 attention、线性注意力和 MLP，仍需由学习者结合任务、显存和消融解释。
