# 学习进度与实验事实记录

> 学习与环境核验已开始，尚未运行实验。由学习者与 Codex 在每次会话结束时更新。不要把计划当完成记录。

## 当前状态

- 当前章节：C00（学习中）
- 当前唯一目标：为真实小模型 SFT 建立可复现环境，同时保持模型下载、模型推理、随机模型训练与真实模型训练的证据边界。
- 下一步：学习者亲自检查一个 Qwen3.5-0.8B SFT 样本的 token、labels 和 assistant mask，确认监督区域后再运行真实 GPU optimizer step。
- 最近一次会话：2026-09-14，Codex 按学习者授权完成 C00 SFT 环境、固定模型下载和 smoke test 准备。
- 已确认背景：过去自写过 Transformers Trainer SFT；RTX 5070 Ti 16GB。
- 已完成环境核验与独立项目环境：WSL2/Arch、uv Python 3.12.13、PyTorch/Hugging Face/TRL 栈和 RTX 5070 Ti 可用；系统 Python 未改动。
- 存储约定：环境使用 `uv` 管理；正式仓库已复制到 `/mnt/d/llm-posttraining-lab`，后续虚拟环境、模型权重和模型缓存均放到 D 盘。C 盘源副本暂时保留以便回退。
- 已完成实验：仅 C00 smoke；无正式数据训练或效果实验。
- 当前阻塞：无；真实 Qwen GPU backward/optimizer step 尚未授权和执行，优化 attention kernel 尚未安装且不是当前前置条件。

## 章节状态

| 章节 | 主题 | 学习状态 | 执行程度 | 效果证据 | 证据/缺口 |
|---|---|---|---|---|---|
| C00 | 全貌与环境 | LEARNING | SMOKE_ONLY | NOT_EVALUATED | HF/TRL 栈和随机微型 LoRA SFT 1 step 已通过；Qwen3.5 0.8B Base forward、0.8B/2B 生成及真实 0.8B LoRA 注入/保存/重载通过；真实 Qwen backward/optimizer NOT RUN；知识地图非独立完成 |
| C01 | 任务、工具、verifier、eval | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 无 |
| C02 | 合成数据工厂 | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 无 |
| C03 | Tool-use SFT | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 无 |
| C04 | DPO 对照分支 | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 可按主计划延后训练，不能自动认定通过 |
| C05 | 单轮 GRPO | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 无 |
| C06-A | 多轮 rollout 与环境 | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 完成此项不代表完成 RL |
| C06-B | 多轮 RL 参数更新 | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 无 |
| C06-C | verl 组件映射 | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 阅读映射必修；实际迁移可选 |
| C07 | 知识/Skill 内化与修复 | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 无 |
| C08 | 复现、答辩、迁移 | NOT_STARTED | NOT_RUN | NOT_EVALUATED | 无 |

状态说明见 `AGENTS.md`；没有效果对照时使用 NOT_EVALUATED。PASSED 必须有本章要求的证据。

## 授权与资源预算

- 付费 API：未授权；费用上限 0；只使用 mock。
- 模型下载：2026-09-14 已授权并完成三个固定 Qwen3.5 snapshot；后续新增模型仍需说明用途与体积。
- 环境安装/修改：2026-09-14 已授权在 D 盘项目 `.venv` 中安装/升级 SFT 准备所需依赖；不得据此修改系统 Python、驱动或 C 盘旧仓库。
- 环境管理器：统一使用 `uv`；创建环境前仍需确认 Python/依赖方案和安装预算。
- 存储位置：正式仓库为 `/mnt/d/llm-posttraining-lab`；虚拟环境、模型权重及 Hugging Face 缓存均使用 D 盘。C 盘源副本暂时保留，不在其中继续开发或下载模型。
- GPU 训练：未授权；先明确配置、步数、停止条件和日志位置。
- 本地轻量 CPU 测试：按 `AGENTS.md` 检查无外部副作用后执行。
- 数据边界：仅虚构任务与公开/获授权数据；不得擅自使用公司私有资料。

后续明确授权时记录日期、范围、请求数/token/费用或训练步数上限，不记录密钥。既有授权范围内不重复询问。

## 环境与版本（已部分检查）

| 项目 | 实际值/证据 |
|---|---|
| 操作系统/运行方式 | Arch Linux on WSL2；`uname -a`、`/etc/os-release`，2026-09-13 |
| 工作目录/项目根 | 正式工作目录为 `/mnt/d/llm-posttraining-lab`；逐文件 SHA-256 对比迁移一致；当前不是 Git 仓库；C 盘源副本暂留 |
| Python 可执行路径与版本 | 系统 `/usr/sbin/python` 为 3.14.5 且无 pip；uv 0.11.19；已存在 uv 管理的 CPython 3.12.13，拟用于项目环境 |
| NVIDIA 驱动/GPU 检测 | `nvidia-smi`：RTX 5070 Ti 16303 MiB；KMD 610.47；CUDA UMD 13.3 |
| PyTorch/CUDA/设备能力 | `torch==2.14.0+cu130`；`torch.version.cuda=13.0`；CUDA 可用；RTX 5070 Ti CC `(12,0)`；wheel 含 `sm_120`；FP32/BF16 forward/backward finite；峰值约 65.38 MiB |
| Transformers / TRL / PEFT / Datasets | Transformers 5.17.0；TRL 1.13.0；PEFT 0.20.0；Datasets 5.0.1；Accelerate 1.15.0；pytest 9.1.1；未安装 bitsandbytes |
| 锁定依赖文件 | `requirements/C00-torch-cu130.lock`；`requirements/C00-hf-stack.lock` |
| 模型 ID / revision | `Qwen3.5-0.8B-Base@dc7cdfe…`、`Qwen3.5-0.8B@2fc0636…`、`Qwen3.5-2B@15852e8…`；均经 `hf cache verify`；缓存仅在 D 盘 |
| tokenizer / chat template 版本 | 三个 snapshot 自带 tokenizer；Base 无 chat template，0.8B/2B 后训练版有；本地 processor 使用 `enable_thinking` 参数 |
| 前向/反向/保存重载 | 随机微型模型 CPU forward/backward、TRL SFT optimizer step、LoRA 保存重载通过；真实 Qwen 仅 forward/generation 和未训练 LoRA 注入/保存/重载，backward/optimizer NOT RUN |
| 峰值显存与测量方式 | `torch.cuda.max_memory_allocated/reserved`：0.8B Base forward 1687/1704MiB；0.8B generation 1688/1704MiB；2B generation 4282/4288MiB |

## 数据与评测登记

- task schema / tool schema 版本：未定义。
- split 生成方式与随机种子：未定义。
- train / dev / test-IID / test-OOD 样本数：未生成。
- 数据族隔离和重复检测：未检查。
- test 封存与 hash：未建立。
- 已使用 test 进行调参：未使用（尚无 test）。一旦使用，记录并降级为 dev，重新建立封存集。
- verifier 反例测试：未实现。
- 教师来源、版本、参数与过滤日志：未产生。

## 薄弱点与复习队列

| 知识点 | 上次错误/证据 | 帮助等级 | 下次变式题/复查时机 | 状态 |
|---|---|---|---|---|
| DPO 固定偏好对与 GRPO 在线 rollout | 初次把“多种正负轨迹”同时作为两者切换条件；见 `reviews/2026-09-13_C00_initial.md` | 概念提示 | 审查 C00 map 时换情境复查 | 待复查 |
| GRPO 组内 advantage | 会计算组内差值，但未识别全局均值会错误改变更新方向 | 明确答案 | 使用不同组均值和尺度的新题复查 | 待复查 |
| CUDA 环境分层验证 | 不了解 `nvcc` 与 wheel runtime/kernel smoke test 的区别 | 明确答案 | 获授权建立环境后独立设计 smoke test | 待复查 |
| CUDA smoke test 证据边界 | 无法解释 `cuda_available`、`sm_120`、实际 forward/backward 与完整模型链路的不同证据层级；见 `reviews/2026-09-14_C00_demo.md` | 明确答案 | 用新的 kernel 执行失败情境复查 | 待复查 |
| 独立评测指标 | 能选择 GRPO，但无法定义 task success 与 invalid tool call rate | 明确答案 | C00 map 审查时要求写清分子/分母 | 待复查 |
| IID/OOD、评测污染与归因 | 能正确计算 task success/loop rate 并识别工作流组合 OOD；但曾把同 500 case 的系统结果直接归因为 Qwen，且混淆 OOD/OOV；见 `reviews/2026-09-14_C00_eval-metrics.md` | 第 2、3 题答后明确答案；第 4 题无提示 | 独立设计先切分、后抽取代码库的 train/dev/test-OOD 方案 | 待复查 |
| 全同 reward 与探索 | 原题回答“不知道”；看过答案后在新情境中独立选择 SFT bootstrap，并指出出现成功 rollout 后再切换 GRPO；见 `reviews/2026-09-14_C00_demo.md` | 原题明确答案；变式无提示 | 后续补查组内奖励差异、采样成本和 verifier 防投机边界 | 初步通过 |

## 实验登记

| 实验 ID | 章节/父 checkpoint | 配置与报告路径 | 执行程度 | 真实结果与样本数 | 下一步 |
|---|---|---|---|---|---|
| C00-TORCH-SMOKE-001 | C00 / 无模型 | `reports/environment.md`；`requirements/C00-torch-cu130.lock` | SMOKE_ONLY | CUDA available；CC 12.0；FP32/BF16 forward/backward finite；峰值 65.38 MiB | 学习者解释证据边界；第二阶段另行授权 |
| C00-HF-STACK-001 | C00 / 随机微型 GPT-2 配置 | `reports/environment.md`；`requirements/C00-hf-stack.lock` | SMOKE_ONLY | CPU SFTTrainer 1 step；loss 2.2742；2 个 LoRA 张量变化；adapter 保存；样本数 2 | 用真实 Qwen 前先由学习者检查 token/labels/mask |
| C00-QWEN-INFER-001 | C00 / 三个固定 Qwen3.5 snapshot | `reports/environment.md`；`reports/model_selection_and_memory.md` | SMOKE_ONLY | 3 个 snapshot checksum PASS；0.8B Base logits/causal loss forward、0.8B/2B 短生成 PASS；无 backward | 真实训练前先做样本监督区域验收 |

## 会话记录索引

| 日期/会话 ID | 本次目标 | 笔记/审查/实验路径 | 独立完成与帮助 | 未完成项 | 下次唯一任务 |
|---|---|---|---|---|---|
| 2026-09-14 / C00-demo | 建立 SFT/DPO/GRPO 数据—反馈—评测地图 | `notes/C00_map.md`；`reviews/2026-09-14_C00_demo.md` | Codex 完整实现地图；学习者独立完成路径选择变式题 3/4 | uv 环境与 CUDA/模型 smoke test；其他薄弱点待复查 | 核验并确认 D 盘 `uv` 环境方案 |
| 2026-09-14 / C00-env-plan | 核验 D 盘 uv/PyTorch 环境方案 | `reports/environment.md`；官方 PyTorch/NVIDIA/uv 文档 | 环境事实为只读检查；方案由 Codex 编写 | PyTorch CUDA、模型与 LoRA 均未运行 | 授权后执行第一阶段 torch + CUDA smoke test |
| 2026-09-14 / C00-torch-smoke | 建立最小 PyTorch CUDA 能力证据 | `reports/environment.md`；`requirements/C00-torch-cu130.lock` | Codex 执行获授权安装与 smoke test；非学习者独立实现 | Transformers/TRL、模型 forward/backward、LoRA 保存重载均 NOT RUN | 学习者解释 smoke test 的证据边界 |
| 2026-09-14 / C00-eval-metrics | 定义端到端指标并区分 IID/OOD 与评测污染 | `reviews/2026-09-14_C00_eval-metrics.md`；`notes/C00_map.md` | 学习者独立完成指标计算和工作流 OOD 变式；污染边界及 IID/OOD 定义经明确讲解 | 尚不能独立设计无泄漏的任务族切分；CUDA 证据边界暂缓 | 独立写出先切分、后抽取代码库的最小 train/dev/test-OOD 方案 |
| 2026-09-14 / C00-sft-env-prep | 提前准备真实 SFT 所需软件、模型和可复现入口 | `reports/environment.md`；`reports/model_selection_and_memory.md`；`requirements/C00-hf-stack.lock`；`scripts/env.sh` | Codex 在学习者授权下安装、下载和执行 smoke；不是学习者独立实现或真实训练效果 | 真实 Qwen backward/optimizer/LoRA 重载 NOT RUN；未安装优化 attention kernel | 学习者检查一个真实 SFT 样本的 token/labels/assistant mask |

## 决策与偏离计划记录

| 日期 | 决策及理由 | 对验收的影响 | 后续补齐条件 |
|---|---|---|---|
| 2026-09-13 | 使用 `uv` 管理项目环境；正式仓库复制到 `/mnt/d/llm-posttraining-lab`，后续虚拟环境、模型权重与缓存均放 D 盘 | 不改变 C00 验收标准；D 盘副本经逐文件 SHA-256 对比一致，C 盘源副本暂留以便回退 | 后续所有学习文件以 D 盘仓库为准；删除 C 盘副本需另行确认 |
| 2026-09-14 | 为覆盖 Base/后训练起点和 16GB 训练迁移，下载 0.8B Base、0.8B 后训练版及 2B 后训练版，暂缓 2B Base | 三个模型只提供环境与推理 smoke 证据；不等于训练或效果通过 | 先在 0.8B 完成 SFT/评测闭环，再决定是否需要 2B Base 或更大模型 |

## 最终能力结论（暂不填写结果）

- 基础学习状态：未验收。
- 单轮 RL 实训状态：未开始。
- 多轮 RL 实训状态：未开始。
- 效果提升证据：无。
- 待团队算力验证事项：尚未识别。
