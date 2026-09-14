# LLM 后训练学习计划：从 SFT 到多轮 Agentic RL

> 版本：v1.0 · 资料核验：2026-09-13  
> 使用方式：放进本地 `llm-posttraining-lab/`，由 Codex 按章节担任助教、提问者和代码审阅者。  
> 面向背景：约两年前使用 Transformers Trainer 自写过 LLM SFT；现在需要补齐数据构造、工具使用、多轮 RL 和知识/Skill 内化。  
> 本地条件：RTX 5070 Ti，16GB 显存；可以获得 GPT / Opus API，但没有现成训练数据。  
> 本包是课程与验收规范，不是已经实现或在该显卡上测试过的训练仓库。章节中的脚本、数据和报告是待完成的作业。

---

## 先读这一页

**你的目标不是“把所有后训练算法都学一遍”，而是独立完成一个可检查、可复现的数据—训练—评测闭环。**

主线：

```text
任务与评测 → 真实工具环境 → 教师生成且可验证的数据 → Tool-use SFT
                                                       ↓
                                              单轮 GRPO → 多轮 Agentic RL
                                                       ↓
                                          失败分析、修复数据、再训练与验收
```

DPO 是短小的对照分支，不是进入 GRPO 的必要前置训练；知识内化单独设置对照实验；verl 是后期理解和迁移目标，不是本地入门的安装门槛。

建议按 **6～8 周、累计约 60～90 小时**安排学习。这是学习预算，不是保证完成的时间；以验收为准，不按日历强行跳章。忙时只做必修项，扩展项以后再补。

每章都要留下三种证据：**你自己的解释、实际代码/测试、真实实验记录**。看完视频只代表完成阅读，不代表章节通过。

### 文档导航

| 位置 | 内容 |
|---|---|
| 第一部分 | 最终能力、学习边界、技术栈与显存策略 |
| 第二部分 | 贯穿全程的小项目、数据与评测规范 |
| 第三部分 | C00～C08 各章资料、目标、作业、验收 |
| 第四部分 | Codex 教学流程、复习与代码审查标准 |
| 第五部分 | 资料总表，含五道口纳什视频入口 |
| 第六部分 | 第一次启动、日常学习和最终验收指令 |

---

# 第一部分：你要建立什么能力

## 1.1 最终应能独立完成的事情

| 能力 | 最终证据，而不是口头宣称 |
|---|---|
| 选择训练路径 | 能说明一个失败更适合改数据、改环境、SFT、偏好学习还是 RL，并给出排除其他方案的理由 |
| 生产训练数据 | 从任务种子生成可执行轨迹；保留来源、教师版本、工具输出、过滤原因和成本 |
| 正确完成 SFT | 能检查 chat template、label shift、loss mask、截断、LoRA 保存与重新加载 |
| 理解偏好学习 | 能构造可信 chosen/rejected，解释 DPO 的相对偏好目标与参考策略 |
| 完成一次可验证 RL | 能把 reward → advantage → logprob → loss → 参数更新逐段对应起来 |
| 实现多轮训练闭环 | 能区分模型动作与环境观察，处理工具失败、终止、环境重置和预算限制 |
| 检验知识/Skill 内化 | 能用有/无指导、新任务组合、新规则等对照，避免把提示词收益误报为参数收益 |
| 判断训练是否有效 | 除训练 loss/reward 外，能报告独立成功率、失败类型、调用成本和不确定性 |

**学习通过、工程跑通、效果提升是三个不同结论。** 正确实现后没有提升，也可以形成合格的学习报告；不能为了“毕业”修改测试答案或隐藏负结果。

## 1.2 先修正几种容易误导学习的理解

- CPT/Mid-training、SFT、RL 不是严格互斥的“知识、行为、策略”三格。SFT 也能注入知识，RL 也可能改变输出习惯。选择依据是数据形态、反馈信号、成本与目标，不是术语本身。
- 能完成 `LLM → tool → LLM` 不代表做过 Agentic RL；还要有学生策略产生轨迹、计算奖励并更新参数的过程。
- 能用闭源 API 生成答案，不等于能直接进行任意形式的 token 分布蒸馏；要核验接口实际提供什么。
- 16GB 适合缩小实验范围，不代表某个模型尺寸的所有 SFT/DPO/RL 配置都能运行。
- 一次训练 reward 上升，不证明泛化提升，更不证明能够替代更强的模型。

术语对照阅读：[TRL 数据类型][R03]、[DPO 原论文][R15]、[DeepSeekMath][R16]、[On-policy Distillation 原论文][R19]。这些是概念依据，以下项目设计和学习门槛是本计划的建议。

## 1.3 技术栈：只保留一条主线

```text
Python / PyTorch
    ├── Transformers：模型、tokenizer、生成与模板
    ├── Datasets：数据读取、变换与导出
    ├── PEFT：LoRA；必要时加入量化训练
    └── TRL：SFT、DPO、GRPO

本项目自写的轻量部分：
任务生成器、工具环境、教师适配器、验证器、轨迹记录、评测器

后期选修：
vLLM → verl 的 rollout / AgentLoop / 多卡训练组织
```

暂不同时学习多套配置式训练框架，不先上分布式，不把浏览器、任意 Shell、真实写操作接进实验。先理解数据契约和训练路径，再考虑框架迁移。

## 1.4 本地显存与兼容性策略

首选 `Qwen/Qwen3-0.6B` 做调试，跑通后再考虑 `Qwen/Qwen3-1.7B`。这里的模型是学习用基线，不是在推荐当前最强的小模型。两个模型的官方卡片见 [R09][R09]、[R10][R10]。

| 场景 | 起始方案 | 扩大条件 |
|---|---|---|
| CPU 数据与单测 | 不加载模型 | 工具和验证器测试先通过 |
| 推理 / SFT | 0.6B，LoRA，短序列，micro-batch 1 | 完成前向、反向、保存、重载并测得显存余量 |
| DPO | 同一 SFT checkpoint，短 completion，小批量 | 确认 reference 的来源与显存开销 |
| 单轮 GRPO | 0.6B，LoRA，短生成，分组采样 | 有非零奖励差异、无异常截断、吞吐可接受 |
| 多轮 RL | 先限制 2 次工具调用，再逐步到 4～5 次 | 每条轨迹都可追踪，环境重置和 token 对齐测试通过 |
| 1.7B / 4B / 多卡 | 非入门必需 | 先完成小模型闭环，再另开环境和实验分支 |

初始实验建议：SFT 总序列长度从 512～1024 起，单轮生成从 128～256 tokens 起；多轮根据实际轨迹长度重新配置，不能硬截断完整工具协议。可从 LoRA rank 8 或 16 开始。**这些只是试跑起点，不是显存占用保证。**

先尝试较简单的 BF16 LoRA；只有显存确实成为瓶颈且相关内核可用时才加 QLoRA。量化并不会消除激活、KV cache 和 rollout 的显存占用。参考 [PEFT 量化指南][R08]、[TRL 显存指南][R11]。

RTX 5070 Ti 在 NVIDIA 官方表中属于 Compute Capability 12.0；PyTorch 2.7 发布说明记录了 Blackwell/CUDA 12.8 支持的引入，但**不要因此固定安装历史版 2.7，也不要照抄旧视频环境**。由 Codex 根据当前官方安装页与本地驱动选择兼容组合，试跑成功后锁定版本。参考 [NVIDIA][R23]、[PyTorch 历史说明][R24]、[当前安装页][R22]。

若本机是 Windows，可优先评估独立 WSL2/Linux 学习环境；已有可用环境则先检查，不自动迁移或重装。不要改动已有工作环境。首次不安装 FlashAttention、vLLM、verl 等额外复杂依赖；优先采用框架默认/SDPA 路径，按实际报错判断。

### 降配顺序

先保存 OOM 日志和配置，再依次减少序列长度、生成长度、同时生成的数量、micro-batch，启用已验证的 checkpointing，必要时减小模型。GRPO 减少组大小时仍需保留有效组间比较，并重新检查 batch/group 的约束。

**不能用缩短到截断答案、把 GRPO 组大小变成 1、关闭所有必要训练项等方式“伪跑通”。** CPU 可完成数据和数学单测，但 CPU 单测不能冒充 GPU 训练。多轮 RL 确实受限时，记录 `BLOCKED` 与迁移需求，不标记为已完成。

---

# 第二部分：用一个小项目贯穿全程

## 2.1 项目：Mini Record-and-Rule Agent

建立一个完全本地、可重置、无外网副作用的小型数据查询和规则计算环境。

只提供三个白名单工具：

```python
lookup_record(record_id: str) -> dict
lookup_rule(rule_id: str) -> dict
calculate(op: str, a: float, b: float) -> dict
```

`calculate` 只允许明确列举的运算，例如 add/subtract/multiply/divide；不对模型提供的字符串使用 Python `eval`，也不提供任意文件或 Shell 访问。

一个任务例子：

```text
用户：请计算记录 R17 在它适用的规则下的最终值。

lookup_record("R17")
→ {"unit_value": 37, "count": 6, "rule_id": "K12"}

lookup_rule("K12")
→ {"description": "先计算 unit_value × count；count ≥ 5 时再减 3"}

calculate("multiply", 37, 6) → 222
calculate("subtract", 222, 3) → 219

最终输出：{"answer": 219}
```

这是虚构任务，不涉及真实财务或业务决策。记录值在不同任务实例中变化；ID 应是不可推断答案的标识。任务答案由独立 Python oracle 计算，不交给模型自行声称正确。

前期所有规则来自运行时工具；C07 再引入一小组固定的虚构规则做知识内化实验。不要把动态记录值当成待背诵知识。

## 2.2 难度递进

| 等级 | 任务特点 | 学什么 |
|---|---|---|
| L0 | 问题已给全部信息，可直接回答 | 不应乱调用工具；用于最小 SFT/单轮 RL |
| L1 | 查询一个记录字段 | 工具选择、参数、结果理解 |
| L2 | 先查记录再查规则，完成计算 | 多步依赖、观察驱动的下一步动作 |
| L3 | 记录不存在、必要信息缺失、可恢复错误 | 报错处理、澄清、终止，而不是编造结果 |
| L4 | 未见过的规则组合、新 ID、新工具 schema 表述 | 泛化与 Skill 迁移 |

先做 L0～L2。L3 的错误必须可复现，例如固定任务在指定调用首次返回一次临时错误；不要让不可控随机失败淹没训练信号。

“多轮”先指 **模型—工具—模型** 的多轮交互。真实用户会反复补充要求的多轮对话，是另一个需要用户模拟器和额外评测的扩展，不混进第一个实验。

## 2.3 数据和测试先分开

建议从下列规模起步，数量不是硬性业绩指标：

| 数据 | 初始规模 | 用途 |
|---|---:|---|
| train 任务实例 | 50 | 调试任务、采集教师轨迹；后续扩至 200～500 条合格轨迹 |
| dev | 20～50 | 日常调试、选 checkpoint、修改提示词 |
| test-IID | 50～100 | 同任务分布、不同记录/随机种子，最终评估 |
| test-OOD | 50～100 | 预留组合/规则变化，最终泛化评估 |

先固定任务家族、规则组合和实体的划分，再扩写语言表达。IID 可以共享任务模板，但不能共享同一底层任务实例；OOD 要明确究竟隔离了什么，不能仅凭“换了一个数字”就叫 OOD。

同一底层任务的改写、教师重试、修复轨迹归为一个 `family_id/instance_id` 簇，不能散落到训练和测试两边。每个数据版本保存 manifest、随机种子、内容 hash 和划分规则。

开发期看 dev；所有候选配置冻结后再比较 test。已根据某个 test 的具体失败调整过数据或提示词，就把它降级为 dev，并另建未见过的测试，不继续宣称独立测试。

## 2.4 教师数据生产流水线

```text
自己定义任务生成规则与 oracle
    → 生成任务与环境快照
    → GPT/Opus 只看任务、可见历史与工具 schema
    → 真正执行教师发出的工具调用
    → 回填真实 observation
    → 教师继续，直到 final / 超时 / 超预算
    → 独立验证最终结果与过程合法性
    → 过滤、去重、分难度、记录来源与成本
    → 导出学生模型所需的数据格式
```

**不让教师一次性编出整段 tool result，再把它当成真实执行数据。** 教师可帮助改写问题，但环境状态和标准答案由程序决定。

建议先做 20 条小批次，人工复核全部，再扩大到 200～500 条合格轨迹；只有评测显示覆盖不足才进一步扩到 1k～2k。限制最大调用数、重试数、并发和单任务输出长度；预算未授权时只使用 mock 教师。

教师模型的最终答案、结构化动作和必要的简短解释就足以构造基础样本，不把获取隐藏思维链当成前提。仅使用获得授权、符合提供方与团队约束的数据和接口；公司数据不自动进入个人机器或外部 API。

## 2.5 统一内部记录，分别导出训练格式

内部轨迹至少包含：

```text
task_id / instance_id / family_id / split
environment_version / snapshot_id / random_seed
messages / tools / tool_execution_log
teacher_provider / teacher_model / generation_config
verifier_version / verified_success / reject_reason
input_tokens / output_tokens / retries / elapsed_time / cost_if_known
```

不知道价格就记录 token、调用数与“成本未配置”，不能把未知价格填成 0。

训练导出和审计记录分开保存。`expected_answer`、隐藏 oracle、测试标签、教师批语不能被错误拼进学生 prompt。

### SFT：一个完整的最小结构示例

以下是仓库内规范化示例，不是某家闭源 API 的原始返回。不同 API 的 arguments 字符串、call ID 等先通过适配器规范化；额外 ID 记录可保留在原始日志中。

```json
{
  "messages": [
    {"role": "user", "content": "返回记录 R17 的 count，只输出 answer 字段。"},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "lookup_record",
            "arguments": {"record_id": "R17"}
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "lookup_record",
      "content": "{\"unit_value\":37,\"count\":6,\"rule_id\":\"K12\"}"
    },
    {"role": "assistant", "content": "{\"answer\":6}"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "lookup_record",
        "description": "按 ID 返回本任务环境中的记录。",
        "parameters": {
          "type": "object",
          "properties": {"record_id": {"type": "string"}},
          "required": ["record_id"],
          "additionalProperties": false
        }
      }
    }
  ]
}
```

SFT 工具数据的 `messages`、`tool_calls`、`tool` role 与 `tools` 字段依据 [TRL 数据格式][R03]。字段正确只是第一步，还要检查模型模板最终渲染成什么 token。

DPO 使用 `prompt/chosen/rejected`；第一版只比较**相同前缀下的一个 assistant 回合**，不要未经设计直接把带不同工具观察的整条轨迹塞进去。GRPO 数据从 `prompt` 加环境元数据开始，完成轨迹和奖励在训练时产生；不等于不需要任务、环境和验证器。[R03][R03]、[R12][R12]、[R17][R17]

## 2.6 指标定义先固定

| 指标 | 本项目定义 |
|---|---|
| Task success | 在预先固定预算内，最终结果满足独立 oracle，且没有禁止动作；异常、超时也计入总任务数 |
| Invalid tool call rate | schema/参数/工具名不合法的调用数 ÷ 总调用数；无调用时报告 N/A 或按文档约定处理 |
| Tool cost | 每任务工具次数、生成 token 数、耗时；成功任务与全部任务分别统计 |
| No-tool correctness | L0 类问题的正确率与不必要调用比例 |
| Recovery success | 可恢复错误子集中的成功率；单独列样本数 |
| IID / OOD success | 分别报告；写清 OOD 划分方式 |
| Sampled success@k | 对每任务完整独立 rollout k 次，至少一次成功的比例；成本通常也随 k 变化 |

不要把采用了复杂无偏估计的论文 `pass@k` 与这里简单的 `success@k` 混写。`success@1` 可以是固定采样配置下的一次尝试，不自动等于贪心解码；报告 temperature、种子和终止规则。

若只评 50 个任务，一个任务就改变 2 个百分点。报告正确数/总数；比较模型时尽量用相同任务做配对比较，在需要主张稳定增益时补重复种子或配对 bootstrap。小样本结果不使用“显著领先”等未经统计支持的表述。

---

# 第三部分：按章节执行

## 总体节奏

| 周次建议 | 章节 | 主要产出 |
|---|---|---|
| 第 1 周 | C00 + C01 | 环境检查、知识地图、任务与评测器 |
| 第 2 周 | C02 + C03 前半 | 数据流水线、模板与 mask 检查 |
| 第 3 周 | C03 后半 + C04 | SFT 基线；短 DPO 对照 |
| 第 4 周 | C05 | 数学单测与真实 GRPO 小实验 |
| 第 5 周 | C06 | 多轮 rollout、工具 RL、verl 架构阅读 |
| 第 6 周 | C07 + C08 | 知识/修复 SFT 对照、最终复现报告 |
| 第 7～8 周 | 弹性缓冲 | 环境排障、未通过章节、可选扩展 |

---

## C00｜恢复手感：后训练地图与环境体检

**建议投入：4～6 小时。前置：Python/PyTorch 基础。**

### 学习目标

能把 SFT、DPO、RLHF、RLVR、GRPO、蒸馏、Agentic RL 放在同一张“数据—目标—更新—评估”图上；确认本地具备最小训练能力，而非只确认能加载模型。

### 按顺序读

| 优先级 | 资料 | 只需读到什么程度 |
|---|---|---|
| 必修 | [Smol Course：课程入口][R01] | 浏览 Instruction Tuning、Preference Alignment、Evaluation，找回概念；不必完成全课程 |
| 必修 | [TRL：数据类型][R03] | 画出 SFT/DPO/GRPO 输入差别 |
| 选读 | [Stanford CS336 2026][R02] | 找到 Data 与 Mid/Post-training、RLVR 相关课；本章先浏览，不看完整学期 |
| 操作参考 | [PyTorch 安装][R22]、[Qwen 0.6B 模型卡][R09] | 对照本地系统与实际安装版本 |

### 你要做

编写或补完 `scripts/preflight.py`：记录 Python、操作系统、驱动、PyTorch/CUDA、GPU 型号、compute capability、实际包版本和可用显存。不要打印密钥或完整环境变量。

分步完成 CUDA tensor 运算、模型一次生成、一个最小训练 batch 的前向与反向、LoRA 保存与重载。使用独立环境；先不跑正式数据训练。模型和依赖需要下载时先确认路径、磁盘和网络权限。

写 `notes/C00_map.md`，用自己的话回答“没有数据时先做什么”“为什么工具 SFT 先于复杂 RL”“RL 为什么还需要数据”。

### 交付与验收

- `reports/environment.md`、精确版本记录、`notes/C00_map.md`。
- 能说清学习模型 checkpoint 与真正的 pretrained base checkpoint 的区别；报告不要把已后训练 Qwen 简称为未经训练的 Base。
- 前向、反向和重载有实际命令及结果。失败则定位到环境/内核/显存，不自动把问题归结为算法。

**Codex 提问方向：** 数据形态、teacher forcing、LoRA 与全参数训练的差别；根据你的回答补基础，不默认你熟悉所有新术语。

---

## C01｜先建任务与评测，不急着训练

**建议投入：6～8 小时。前置：C00。**

### 学习目标

独立写出可验证、可重置、具有多步依赖的任务；知道“模型看见什么、评测器知道什么”。

### 按顺序读

| 优先级 | 资料 | 重点 |
|---|---|---|
| 必修 | [Transformers：Tool use][R07] | 工具 schema、assistant 调用、工具回填，不把工具输出视为模型生成 |
| 必修 | [TRL：Tool Calling 数据格式][R03] | 原始轨迹到训练字段的映射 |
| 选读 | [CS336：Data 相关课][R02] | 数据来源、去重和划分；不要求实现完整网页清洗作业 |

### 你要做

实现 `src/environment.py`、`src/verifier.py`、`src/evaluate.py` 和对应测试。先用手写正确/错误轨迹测试环境，再接本地模型；模型不会调用工具是可记录的初始结果，不用人工代做来提高基线分数。

完成 L0～L2、至少一种缺失记录任务，以及固定 split manifest。允许不同的有效计算路径，不把唯一教师工具顺序作为唯一正确答案；只有任务确实要求时才考核特定工具。

至少测试：正确答案、错误答案、无效 JSON、缺字段、重复 answer 字段、多个互相冲突的答案、未知工具、除零、缺失 ID、超步数、环境重置。工具返回的字符串中混有指令时，只视为不可信数据。

### 交付与验收

- `data/tasks/`、`data/manifests/`、`tests/test_environment.py`、`tests/test_verifier.py`。
- 已定义测试的正例和反例全部按预期判定；不能靠大模型一句“看起来正确”放行。
- 同一 snapshot/seed 可复现；一个任务的操作不会污染另一任务。
- `reports/baseline_dev.md` 记录未做本项目训练的学生模型在 dev 上的结果，包括 0 分情况。

**Codex 提问方向：** IID/OOD 的区别；如何防止答案泄漏；为什么超时和失败不能从分母剔除。

---

## C02｜从零造数据：教师、执行器、过滤器

**建议投入：6～8 小时。前置：C01。**

### 学习目标

能把一个 GPT/Opus 调用写成可审计的数据生产过程，而不是“问模型生成 5000 条问答”。

### 按顺序读

| 优先级 | 资料 | 重点 |
|---|---|---|
| 必修 | [Self-Instruct 原论文][R14] | 种子、扩展、过滤的总体方法；不照搬特定模型配方 |
| 必修 | [CS336：Data 相关课][R02] | 去重、混合和合成数据；记录哪些可迁移到轨迹数据 |
| 操作参考 | [TRL 数据格式][R03] | 合格原始轨迹如何导出成训练集 |

### 你要做

先写 mock teacher 让流水线离线跑通，再实现一个提供方适配器。通过后才加入第二个提供方；不先写复杂 SDK 抽象。教师每次只生成下一回合，由执行器处理工具，再继续调用教师。

首批生产 20 条，逐条检查。随后按已授权预算扩至 200～500 条验证通过的轨迹。保留失败候选和拒绝原因，禁止把错误样本静默删除后只报告成功数。

实现断点续跑、稳定 task ID、缓存、限流与有限重试。变更模型或 prompt 后要更新缓存 key。不要把 train 与 dev/test 的缓存混在一起。

抽查覆盖：直接回答、单工具、依赖调用、失败恢复、不该调用工具。统计有效率、去重后数量、长度分布、任务家族数、每条合格轨迹消耗。

### 交付与验收

- `src/teacher.py`、`scripts/build_dataset.py`、`data/raw/`、`data/processed/`。
- `reports/data_card.md`：来源、授权边界、schema、split、教师配置、质量与成本。
- 随机抽 20 条合格数据，工具输出都能与执行日志对上；筛选通过不等于所有潜在错误已经消失，仍保留审计。
- 同一任务重试/改写不能跨 split；无法核验的样本标为未验证，不混入“黄金数据”。

**Codex 提问方向：** 教师幻觉与工具真实返回的区别；同一模型生成又评分的偏差；如何判断“数据多”不是“模板重复多”。

---

## C03｜Tool-use SFT：先把训练做正确

**建议投入：10～12 小时。前置：C02。**

### 学习目标

把你以前的 SFT 能力升级为对工具轨迹的训练能力，重点是 token、mask 和推理闭环一致性。

### 按顺序读

| 优先级 | 资料 | 重点 |
|---|---|---|
| 必修 | [Transformers：Chat templates][R05] | role、控制 token、训练与生成前缀的差别 |
| 必修 | [TRL：SFTTrainer][R04] | assistant-only loss、工具调用、PEFT 接入 |
| 必修 | [TRL：训练用 Chat Templates][R06] | assistant mask、模板兼容与多轮前缀保持 |
| 查阅 | [SFTTrainer 源码][R26]、[PEFT 量化][R08] | 只定位数据准备、collator、loss 入口；不通读整个文件 |

### 你要做

先写 `scripts/inspect_tokens.py`：对一条真实工具轨迹输出 token index、token 文本、role、attention mask、label 是否参与 loss。实现可检查输出，而不是凭配置名相信 mask 正确。

本项目默认训练 assistant 的工具动作与最终答案；system、user、tool observation 作为条件输入，不作为 SFT 预测目标。**不计 loss 不等于从上下文删除。** `assistant_only_loss` 依赖模板提供正确 mask；要检查实际版本的行为。[R04][R04]、[R06][R06]

先挑 16～32 条固定短样本做小样本拟合检查，再对完整小数据集训练一个 LoRA。第一版关闭 packing，避免同时调太多变量。保存 adapter、tokenizer/模板配置和模型 revision，再在新进程加载评测。

### 交付与验收

- `scripts/train_sft.py`、`configs/sft.yaml`、`tests/test_loss_mask.py`。
- 每个训练样本有非零有效 label；user/tool/padding 不错误参与 loss；有效 EOS 不因为与 pad 共享 ID 就被无差别屏蔽。
- 能解释 `logits[:, :-1]` 与 `labels[:, 1:]` 的关系，并检查框架是否已完成 shift，避免重复 shift。
- 至少一项本来容易拟合的固定样本任务有明显拟合证据；失败时先查模板、label、学习率和参数是否可训练，而不是马上扩数据。
- `reports/sft_dev.md` 包含初始模型与 SFT 的同配置 dev 对比、失败样例、显存峰值和训练耗时。没有提升也如实写。

**Codex 提问方向：** 工具输出为何不训练但仍保留；packing 和截断各自会引入什么错误；重新加载后结果不同从哪里排查。

---

## C04｜偏好学习：一个短 DPO 分支

**建议投入：4～6 小时。前置：C03。与 C05 可独立推进。**

### 学习目标

理解偏好对的来源、DPO 的相对目标和参考策略。不要因为学过 DPO，就默认最终生产模型要先 DPO 再 GRPO。

### 按顺序读

| 优先级 | 资料 | 重点 |
|---|---|---|
| 必修 | [DPO 原论文][R15] | 从偏好到分类式目标；理解主要公式，不必推完所有附录 |
| 必修 | [TRL：DPOTrainer][R12] | prompt/chosen/rejected、reference、日志 |
| 中文辅助 | [五道口纳什：重新理解 DPO][V03] | 带着“reference 限制什么、隐式 reward 是什么”去看 |

### 你要做

从同一 SFT checkpoint 对相同前缀采样多个候选，构造约 100～300 对。优先用规则正确性或真实工具执行后的结果排序；都错且无法可靠区分、或都对且等价时，允许弃样，不强造偏好。

首版只做单个 assistant 回合的偏好：例如相同任务状态下正确/错误的下一次工具调用，或者相同观测后的正确/错误最终答案。下一步动作的好坏可通过固定后续策略执行验证，但训练 completion 中不要混入环境生成的工具结果。

做一个 DPO 小实验，并与同一父 checkpoint 对比。可选做同预算的“只对 chosen 再 SFT”对照，区分偏好目标与新增好数据的作用。

### 交付与验收

- `scripts/build_preferences.py`、`scripts/train_dpo.py`、`configs/dpo.yaml`。
- `reports/preference_card.md` 说明正负对来源、同前缀检查、长度差异、弃样比例。
- reference 是固定的哪个模型/adapter，必须有记录；不能意外关闭 SFT adapter 后拿另一个基座作 reference。
- 解释 beta 的作用、chosen/rejected 的相对概率，以及为什么 DPO loss 降低不保证环境任务成功。

**Codex 提问方向：** 为什么“不需要单独训练 RM”不等于“不需要偏好信号”；为什么完整多轮轨迹 DPO 需要额外考虑 observation mask。

时间紧时可以把实训标记为 `DEFERRED`，先完成原理与数据作业后进入 C05。不得把延后写成已训练。

---

## C05｜Policy Gradient → PPO → GRPO：从公式走到一次更新

**建议投入：12～16 小时。前置：C03；不要求 C04 训练完成。**

### 学习目标

理解 rollout 来自谁、reward 如何变成 advantage、哪些 token 的 logprob 被优化，以及为什么训练可能没有有效信号。

### 按顺序读

| 顺序 | 资料 | 重点 |
|---|---|---|
| 1 | [OpenAI Spinning Up：Policy Gradient，官方仓库文本][R13] | policy、trajectory、期望回报、log-prob trick；只补相关内容 |
| 2 | [PPO：官方仓库文本][R28] | old/new policy、importance ratio、clip；不要求训练完整 PPO/critic |
| 3 | [DeepSeekMath][R16] | GRPO 如何用组内信息替代单独训练的价值网络 |
| 4 | [TRL：GRPOTrainer][R17] | 生成、奖励、优势、loss 类型、日志与实际实现 |
| 中文辅助 | [五道口纳什：PG loss components][V01] → [策略梯度补充与 GRPO][V02] | 把视频术语对应到自己的张量与配置 |

### 先做 CPU 数学练习

自己写 `exercises/grpo_math.py`，输入手工张量，计算 masked logprob、组内均值/标准差、advantage、ratio 和 clipped surrogate。教学版组内优势可以从以下式子开始：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)
ratio_t = exp(logp_current_t - logp_old_t)
```

标准差使用总体还是样本估计、是否缩放 reward、token/sequence 怎么聚合，均需显式写明；再对照安装版本的实现。不要把这个简化式当成所有 GRPO 变体的完整算法。[R16][R16]、[R17][R17]

至少测试：全 0 reward、全 1 reward、混合正负 advantage、padding、正负优势下的 clip、mask 后分母为 0、旧策略 detach。能区分 `old policy` 和 `reference policy`：前者与采样/重要性比率有关，后者通常用于偏离约束，并非同一个概念。

### 再做真实小模型实验

从本项目 L0 生成可程序验证的短题：把所有输入直接给模型，要求一个结构化最终答案，暂时不调用工具。不要一开始使用复杂数学竞赛题。

先检查基础成功率：任务全部答对或全部答错时，很可能没有有效组内区分，应先调整任务难度或 SFT cold start，而不是奖励随机抖动。

用 LoRA 运行 5～10 个 optimizer steps 的 GRPO smoke test，确认真实更新和保存；然后在预算允许下完成一个较完整小实验。建议组大小从 2 或 4 开始，检查当前 TRL 对 generation batch 与组大小的整除约束，不照抄旧脚本。

### 必须记录

`reward` 的均值与分布、组内 reward 标准差/零方差组比例、有效 completion 长度、截断率、成功率、grad norm、显存、rollout 时间、训练时间。KL、entropy、clip fraction 按当前配置与实现记录；某项未启用就写 N/A，不能捏造一条曲线。

显式记录 loss variant、reward scaling、KL 系数、采样配置和数据复用次数。第一版用默认可工作的生成路径，不强依赖 vLLM。

### 交付与验收

- 数学单测通过；你能手算一组优势，并指出一条轨迹的哪些 token 收到正/负更新信号。
- `scripts/train_grpo.py`、`configs/grpo.yaml`、`reports/grpo_dev.md`。
- 真实学生 rollout → verifier → optimizer update → checkpoint → 独立 dev 评测都有记录；单纯调用 API 生成答案不算完成。
- 至少找出一个奖励漏洞或异常案例，补一个回归测试。不能只靠“输出里包含某数字”判分。

**Codex 提问方向：** 全对/全错组意味着什么；KL 与 entropy 的不同作用；reward 上升但成功率不升时如何诊断；GRPO 不训练 critic 不代表没有 baseline。

---

## C06｜多轮 Agentic RL：让工具环境进入训练

**建议投入：10～14 小时。前置：C01、C03、C05。**

### 学习目标

把单轮“生成答案”扩展成“动作—观察—下一步动作”的可训练轨迹，理解工程正确性与扩展性能的边界。

### 按顺序读

| 优先级 | 资料 | 重点 |
|---|---|---|
| 必修 | [TRL GRPO：Agent Training][R17] | `tools`、环境生命周期、最大工具回合；核验安装版本是否有对应 API |
| 必修 | [verl：Agentic RL][R20] | rollout、训练器、环境与奖励的分工 |
| 必修 | [verl：Agent Loop][R21] | `prompt_ids/response_ids/response_mask` 等数据契约 |
| 中文辅助 | [五道口纳什：AgentLoop 代码串讲][V04] | 对照整体调用链，不照抄视频里的旧行号 |

### C06-A：先完成可检查的多轮 rollout

用 C03 的 SFT 模型完成 `generate → parse → validate → execute → append observation → generate`。设置最大模型回合、工具次数、生成长度与墙钟时间；分别记录正常 final、无效动作、超预算、工具错误和截断。

同一个 GRPO group 应面对相同的初始任务与环境快照，而不是把不同题的轨迹混进同一组。每个 rollout 用独立环境副本，避免共享计数器或错误状态。

保留真实模型生成 token 与位置映射。工具结果是条件输入，不是策略采样动作，不计入 policy loss；不能简单地把全部历史 decode/encode 一遍后假设 token ID 与 old logprob 仍然匹配。模板可能重新渲染历史，因此要做前缀一致性和 token 对齐测试。[R06][R06]、[R21][R21]

### C06-B：再接 RL 更新

当前 TRL 官方文档有工具调用和环境训练入口，本地优先核验并使用它，不因为“Agentic”一词就强行迁到 verl。[R17][R17]

使用同一个冻结的 SFT checkpoint 分出两条对照：`SFT` 与 `SFT → Agentic GRPO`。不要把 C04 的 DPO checkpoint 偷偷换成新的起点，也不要把 L0 单轮结果与 L2 多轮结果直接比较。

奖励先用严格的终局成功 0/1。工具成本先作为独立指标；正确率稳定后才实验成本惩罚，并限制惩罚幅度，使失败但少调用不会比正确完成更优。中间进度奖励必须能独立验证，不能奖励模型声称“任务已完成”。

先 2 个工具步骤、短任务、5～10 次 optimizer update 验证，再扩到 4～5 步。允许同步的最小 loop；异步与推训分离是后期吞吐问题，不是多轮正确性的定义。

### C06-C：verl 只要求读懂，迁移可选

写 `notes/C06_verl_mapping.md`，把自己的环境、生成器、轨迹、奖励、训练器映射到 verl 组件。定位 rollout 返回到 trainer 的接口，说明参数同步、异步请求和并发环境隔离分别解决什么问题。

没有合适算力时不要求在单卡上复制完整分布式配置；留下迁移清单，不把“读懂 AgentLoop”写成“已经完成 verl 训练”。

### 交付与验收

- `src/agent_loop.py`、`tests/test_trajectory_alignment.py`、`tests/test_environment_reset.py`。
- 至少一条依赖两次以上工具调用的完整轨迹，可逐段解释 token/mask/观察/奖励。
- C06-A 与 C06-B 分别记录状态：只完成 A 不算完成多轮 RL 实训。
- C06-B 要有真实多轮 rollout 和策略更新证据；硬件阻塞时记 `BLOCKED`、最小失败配置和待迁移事项，后续章节可继续学习但不抹去缺口。
- `reports/agentic_rl_dev.md` 比较同预算成功率、无效调用、恢复成功率、长度和工具成本。

**Codex 提问方向：** observation 为何不参与策略 loss；工具失败和任务失败的区别；如何检测环境串扰；什么情况下旧轨迹不再能当作当前 on-policy 数据。

---

## C07｜知识与 Skill 内化：先做可验证的小实验

**建议投入：6～8 小时。前置：C03；不要求 C06 已取得性能提升。**

### 学习目标

区分稳定规则、通用做事方法和动态事实；完成一次学生失败驱动的数据修复，并理解它与 token 分布级 OPD 的区别。

### 按顺序读

| 优先级 | 资料 | 重点 |
|---|---|---|
| 必修 | [On-Policy Distillation 原论文][R19] | 学生访问到的状态/序列与固定教师数据的分布差异 |
| 必修 | [TRL DistillationTrainer][R18] | 教师分布、学生生成、词表与接口要求；只读，不强制训练 |
| 复用 | [Self-Instruct][R14]、[SFTTrainer][R04] | 如何把新数据重新放回可靠训练流程 |

### 先定义什么要内化

把 10～20 条虚构但固定的规则写成 `data/knowledge/stable_rules.md`。例如规则 K12 的阈值与扣减逻辑。再写一份通用操作指导：读记录、获取适用规则、校验缺失、计算、按 schema 输出。

动态记录数值、当前工具 schema、权限和环境异常仍由运行时提供。不能删除工具描述后再声称在公平测试“新工具使用能力”。

### 做一个有控制的小实验

选择 C03 的同一父 checkpoint 记为 M0，通过合格规则任务/修复轨迹再 SFT 得到 M1。在相同工具接口、记录数据和任务预算下，对比：

| 条件 | 模型 | 提示中是否给完整规则/通用指导 |
|---|---|---|
| A | M0 | 否 |
| B | M0 | 是 |
| C | M1 | 否 |
| D | M1 | 是 |

另设未见过的任务表达、规则组合，以及规则已更新时必须查询工具的子集。A→B 反映提示辅助；A→C 才较接近参数更新后的差异，仍需控制数据泄漏和其他配置。C 不能单独证明模型学会了任意新规则。

### 完成一次学生失败驱动修复

```text
M0 在 train 任务上的 rollout
    → 根据 verifier 找到失败位置
    → 教师看失败状态，提出下一步修复
    → 真正执行并验证修复
    → 从有效前缀导出修复后的 assistant 目标
    → 混入原有合格数据，再 SFT 得到 M1
```

不要把错误前缀中的全部错误 assistant 动作又当正确目标监督。第一版可训练“失败状态后的正确下一回合”，原有上下文保留但 mask 掉；或者重新生成一条完整、已验证的正确轨迹。留一组原任务回归集检查是否遗忘。

这叫**学生失败驱动的修复 SFT/轨迹蒸馏**，不是自动等同于分布级 OPD。TRL 当前的分布级蒸馏路径需要教师对学生生成上下文给出相应 token 分布，并有词表等要求；单纯让 GPT/Opus 返回答案并不直接提供这个接口。[R18][R18]、[R19][R19]

如果将来要做 OPD，先形成接口能力表：能否评分任意学生指定序列、返回哪些 token 概率、词表/分词如何对齐、top-k/截断近似如何处理、每条序列成本多少。不要把“API 有 logprobs 字段”直接当作所有条件已满足。

### 交付与验收

- `reports/knowledge_ablation.md`、`reports/repair_sft.md`，以及修复数据来源和过滤记录。
- 解释为什么当前优先做小规模规则/修复 SFT，而不是先投入大量 CPT 或强行做闭源 logit KD。
- 能指出一个“看起来内化了、其实只是提示词变强/测试泄漏”的反例。
- 教师修复、验证、再训练的成本分开记录；新数据带来的改善与其他改动不混在一起。

**Codex 提问方向：** 静态知识与动态事实的边界；无指导测试是否公平；response 蒸馏与 token 分布蒸馏的接口差异。

---

## C08｜复现、答辩与项目迁移

**建议投入：4～6 小时。前置：已有真实实验记录；未完成项保持显式。**

### 学习目标

从“我跑过了”变成“别人能检查，我能解释失败和下一步”。

### 资料

不新增主课。只回查本实验用过的官方文档和锁定版本源码，避免最后一章又开启新的论文收藏。

### 你要做

冻结代码、模型、提示词、采样和奖励配置后，用封存 test-IID/test-OOD 做最终评测。相同场景只比较相同父模型、相同任务与预算的分支：

```text
初始学生 M_init → SFT M_sft
                       ├── DPO       （可延后）
                       ├── Agentic GRPO
                       └── 修复/知识 SFT
```

C05 的单轮 GRPO 单列一张表，不与不同任务的多轮结果混用。上述分支是实验，不是强制串联的训练配方。

重新加载 checkpoint 跑一遍 smoke test。若有条件，用干净环境或锁定依赖的另一环境复查；没验证的环境不写“可完全复现”。给出 CPU 测试命令、GPU smoke 命令、完整训练/评测命令，标记哪些实际运行过。

最终报告写清：任务和数据、模型与版本、reward/verifier、关键配置、运行开销、成功率与样本数、至少 5 个失败案例、未完成项、最小下一步。不要只贴一张 reward 曲线。

### 最终验收

Codex 从你的真实代码中选择一段，要求你解释张量形状、数据流或失败原因；再临时改变一个需求，例如增加可恢复错误、改变规则阈值或加入新工具参数，检查你能否自行修改并补测试。

必须能独立回答：

1. 当前主要瓶颈是数据、能力、探索、奖励还是环境？证据是什么？
2. 为什么下一步不一定继续加大模型或改 RL 算法？
3. 哪些本地模块可以迁到团队项目，哪些只适合教学？

`reports/final_report.md` 应分别写出：**基础学习状态、单轮 RL 实训状态、多轮 RL 实训状态、效果证据等级**。一个部分未完成不妨碍报告其他成果，但不能将其一起宣称完成。

---

# 第四部分：让 Codex 当老师，而不是替你交作业

## 4.1 仓库职责

随包的 `AGENTS.md` 放仓库根目录，保存长期教学规则；完整课程保留在本文件，进度放 `PROGRESS.md`，避免每次依赖聊天历史。Codex 官方说明其会读取项目 `AGENTS.md`；首次启动要检查实际加载的指令，已有 override 或上级指令可能影响行为。[官方说明][R25]

建议随着章节逐步创建：

```text
llm-posttraining-lab/
├── AGENTS.md
├── LEARNING_PLAN.md
├── PROGRESS.md
├── README.md
├── templates/
├── notes/                 # 你自己的解释、阅读笔记、错题
├── exercises/             # 自写 loss / 数学练习
├── src/                   # 环境、工具、教师、评测
├── scripts/               # 逐章创建的训练/数据脚本
├── configs/               # 不含密钥
├── tests/                 # CPU 为主；GPU 测试单独标记
├── data/                  # 本地数据，不默认推送到公开仓库
├── runs/                  # checkpoint、日志、原始轨迹
├── reviews/               # 代码审阅、测验记录
└── reports/               # 环境、数据卡、实验与最终报告
```

模板和规则已提供；其余目录/代码是学习过程要创建的，不是假装已有可执行实现。

## 4.2 每个学习单元的固定循环

**诊断 → 阅读 → 自己复述 → 自己实现 → 测验 → 代码审查 → 修复 → 留档。**

每次只学一个小目标。例如 C03 可以拆成“模板”“mask”“训练”“重载评测”四次，不要求一口气完成整章。

开始时，Codex 问 1～2 个先修问题，再给当次最多 2 个资料入口和明确阅读范围。你读完后先口头或文字复述，Codex 根据复述定位缺口，不先贴整篇答案。

写代码时由你负责核心逻辑：任务/验证器、mask、偏好构造、数学 loss 练习、奖励和实验解释。Codex 可以搭 CLI、配置、日志与测试框架；但代写关键函数后必须注明，并通过“你独立修改一个新要求”补验收。

卡住时逐级给帮助：概念提示 → 指出可疑位置 → 伪代码 → 在明确请求后给实现。不把“代码能运行”当成你已经掌握。

## 4.3 章节测验

每章建议 5 题：2 道概念、1 道手算/数据追踪、1 道代码排错、1 道迁移题。**一次一题，你答完再反馈，不提前提供完整答案。** 对非数学章节，把手算题改成 schema、数据流或 split 判断。

每题 0～4 分：0 未掌握；1 只会复述术语；2 在明显提示下完成；3 独立正确；4 能解释限制并举反例。建议总分至少 16/20，且不能在数据泄漏、loss mask、奖励正确性这些核心点上未通过。分数是辅助，不是职业资质认证。

曾经看过答案或得到提示的题，要标注帮助等级，不能按完全独立作答打分。下一次学习开头，用变化后的题目复查上章一个薄弱点；隔一周再做一次跨章迁移，不自动创建日历提醒。

## 4.4 代码审查标准

| 审查层 | Codex 必须检查的点 |
|---|---|
| 数据 | split/family 泄漏，模板重复，oracle 暴露，伪造 tool result，失败样本去向 |
| SFT | tokenization、重复 special token、shift、assistant mask、EOS/pad、截断、可训练参数、保存重载 |
| DPO | 相同前缀、偏好可信度、reference 正确、长度偏差、未把环境观察当动作 |
| RL | old/current/reference 区分、reward→advantage、detach、mask、组内任务一致、全同 reward、采样与 logprob 对齐 |
| 工具/环境 | 白名单、参数验证、timeout、最大步数、状态隔离、可复现错误、禁止任意执行 |
| 实验 | 同父 checkpoint、相同预算、独立 eval、失败分母、版本、成本、结论不超过证据 |

审查输出至少包含：问题严重性、文件与行号、影响、复现方式/测试、最小修改建议、是否实际执行验证。默认先审查不改代码；你明确要求修复后才修改。

不能因为测试没跑就写“应该通过，所以通过”。受环境限制时应写 `NOT RUN`，列出需要你运行的命令。发现重大问题，不允许只改测试期待值或放宽 oracle 来消除失败。

## 4.5 状态与停止规则

章节状态用：`NOT_STARTED / LEARNING / READY_FOR_REVIEW / NEEDS_WORK / PASSED / BLOCKED / DEFERRED`。

C06-A/B 分开。实验执行程度用：`NOT_RUN / SMOKE_ONLY / FULL_RUN`；效果用：`NOT_EVALUATED / NO_GAIN / INCONCLUSIVE / GAIN_WITH_EVIDENCE`。不把一次 smoke test 描述为训练收敛。

出现以下情况先停训练：NaN/Inf、所有训练 label 被 mask、奖励函数反例失效、疑似测试泄漏、工具越权、预算上限到达。先收集最小复现再修复，不让训练继续消耗资源。

---

# 第五部分：资料总表与阅读取舍

## 5.1 官方课程与文档

以下为学习入口与指定阅读范围。不要把整张表当作必须连续读完的书单。

| ID | 资料链接 | 对应章节与范围 |
|---|---|---|
| R01 | [Hugging Face Smol Course][R01] | C00：Instruction Tuning、Preference Alignment、Evaluation；不依赖尚未发布单元 |
| R02 | [Stanford CS336 2026 官网][R02] | C00/C02/C05：Data、Mid/Post-training、RLVR；从官网进入相应讲义或录像 |
| R03 | [TRL Dataset formats][R03] | C00/C01/C02：三类训练数据与 Tool Calling |
| R04 | [TRL SFTTrainer][R04] | C03：mask、packing、PEFT、工具 SFT |
| R05 | [Transformers Chat templates][R05] | C03：模板、生成前缀、特殊 token |
| R06 | [TRL Chat Templates][R06] | C03/C06：训练模板和前缀保持 |
| R07 | [Transformers Tool use][R07] | C01：schema、调用、结果回填 |
| R08 | [PEFT Quantization][R08] | C03 排障：量化加载与可训练 adapter |
| R09 | [Qwen3-0.6B 官方模型卡][R09] | C00：学习用小模型、thinking 模式和模板 |
| R10 | [Qwen3-1.7B 官方模型卡][R10] | 进阶：小模型闭环通过后再扩大 |
| R11 | [TRL Reducing Memory Usage][R11] | C00/C03/C05：显存组成与降配 |
| R12 | [TRL DPOTrainer][R12] | C04：数据、reference 与训练 |
| R13 | [OpenAI Spinning Up：Policy Gradient 原始文本][R13] | C05：PG 推导；使用官方 GitHub，规避文档站访问不稳定 |
| R17 | [TRL GRPOTrainer][R17] | C05/C06：GRPO、日志、Agent Training |
| R18 | [TRL DistillationTrainer][R18] | C07：分布级蒸馏的要求，不强制运行 |
| R20 | [verl Agentic RL][R20] | C06：系统边界 |
| R21 | [verl Agent Loop][R21] | C06：输出契约、mask、组件对应 |
| R22 | [PyTorch 当前安装页][R22] | C00：按本机环境选版本 |
| R23 | [NVIDIA Compute Capability][R23] | C00：核对显卡架构 |
| R24 | [PyTorch 2.7 历史发布说明][R24] | C00：理解 Blackwell 支持来源，不是固定版本配方 |
| R25 | [Codex AGENTS.md 官方说明][R25] | 首次使用：项目教学规则 |
| R26 | [TRL SFTTrainer 源码入口][R26] | C03：定位数据准备与 loss 路径 |
| R27 | [TRL GRPOTrainer 源码入口][R27] | C05/C06：定位 rollout→reward→loss |
| R28 | [OpenAI Spinning Up：PPO 原始文本][R28] | C05：ratio、clipping、old policy |
| R29 | [TRL DPOTrainer 源码入口][R29] | C04：相对 logprob 与 reference |

源码入口链接到主分支，仅用于导航。实际对照时切换到你锁定的 tag/commit；官方在线 `latest` 文档也可能与已安装版本不同。让 Codex 记录版本并查函数签名，不按记忆拼 API。

CS336 不要求从 Transformer、Triton、分布式训练重新学起，也不要求用 16GB 完成官方全部作业。官网课程安排与可播放录像分别核验，不把旧年份视频自动标成 2026。

## 5.2 五道口纳什怎么用

定位为**中文原理与源码辅助课**，不是完整作业体系，也不是 API 的最终权威。

| ID | 视频链接/入口 | 适合什么时候看 |
|---|---|---|
| V01 | [Agentic RL 01：PG loss components，dual-clip，entropy，KL][V01] | C05 补 PG 基础后，理解 loss 组件 |
| V02 | [Agentic RL 03：策略梯度补充，GRPO loss 分析，优势标准化][V02] | C05 手写数学练习前后 |
| V03 | [Agentic RL 11：重新理解 DPO][V03] | C04，配合原论文和 TRL 文档 |
| V04 | [Agentic RL 13：verl infra AgentLoop 代码串讲][V04] | C06 自己跑过多轮 loop 后 |
| V05 | [veRL：核心强化学习算法，GRPO、RLOO、REINFORCE++][V05] | C05/C06 选看，比较 baseline 设计 |

另一个相关主题是“MultiTurn Tool Use / Coding Agent SFT / Cold Start for RL”。本次未取得足够可靠的一手元数据，不写未经确认的 BV 号；可用这个 [B 站关键词搜索入口][VSEARCH] 找到本人视频，确认作者、标题和可播放情况后再加入你的学习记录。

**核验范围：** 上表视频标题/链接依据公开搜索索引核对；B 站详情页直接抓取出现 412，未逐段观看视频，不能保证完整内容免费、配套仓库公开或代码适配当前版本。部分索引说明配套代码有会员/私有仓库限制；本计划不依赖购买或获取这些代码。

看视频时只记录四件事：讲了什么对象、关键公式/接口、对应自己哪段代码、一个仍不理解的问题。看完就做本章小实验，不沿推荐视频继续无限刷课。

## 5.3 论文只保留四篇主阅读

| ID | 原论文 | 本计划关注点 |
|---|---|---|
| R14 | [Self-Instruct][R14] | 数据种子、合成、过滤；C02 |
| R15 | [Direct Preference Optimization][R15] | 偏好目标与参考策略；C04 |
| R16 | [DeepSeekMath][R16] | GRPO 核心思路；C05 |
| R19 | [On-Policy Distillation of Language Models][R19] | 学生分布上的反馈与蒸馏；C07 |

R1、DAPO、Dr.GRPO 等留到你能解释自己 GRPO 曲线之后再选读；第一轮不为“追新”加入更多训练算法。

---

# 第六部分：具体怎么执行

## 6.1 第一次启动

把本包内容放进本地 `llm-posttraining-lab/`。已有同名文件先合并，不覆盖已有个人规则。进入该目录启动 Codex；从其他工作目录启动时，先核对它实际识别的项目根目录和指令。

把下面这段发给 Codex：

```text
请把这个目录当作我的 LLM 后训练学习仓库。
先读取 AGENTS.md、LEARNING_PLAN.md 和 PROGRESS.md，确认你加载了哪些教学规则。

我有过去用 Transformers Trainer 做 SFT 的经验，但后训练/RL 知识需要逐步补齐。
本机是 RTX 5070 Ti 16GB，已有工作环境不要改动。
你是我的老师与代码审阅者，不是替我一次性生成整个项目的外包工程师。

本次只开始 C00：
1. 先用少量问题诊断我的现有理解，不一次给出全部答案。
2. 检查当前操作系统、Python 环境、GPU/驱动及可用工具。
3. 给我本次最多两个阅读入口、阅读范围和一个最小作业。
4. 可以创建必要的空目录和记录模板，但不要提前实现后续章节的核心代码。
5. 未经授权，不安装大批依赖、不下载大模型、不调用付费 API、不启动长训练。
6. 用 PROGRESS.md 保存进度，以后根据记录继续，而不是依赖聊天记忆。

请先做诊断和环境检查计划。
```

第一次不需要配置 GPT/Opus 密钥，也不需要安装 verl。先确认知识起点与机器，再决定最小安装方案。

## 6.2 每天开始学习

```text
读取 PROGRESS.md 和最近一次 reviews/ 记录，继续当前章节。
先用一道变式题复查上次薄弱点；然后只安排一个本次可完成的小目标。
先告诉我读什么、产出什么、怎样验收，不要直接替我写核心答案。
```

## 6.3 阅读结束，请它测验

```text
我已经完成 CXX 的指定阅读。请按教学规则提问验收。
一次一道题，先等我作答再反馈；不要提前给参考答案。
请包含概念、数据/公式追踪、排错和迁移问题。
记录我用了哪些提示，并把错题和需要补学的点写入 reviews/ 与 PROGRESS.md。
```

## 6.4 代码完成，请它审查

```text
请审查我在 CXX 写的代码。先读本章验收标准，再检查 git diff 和相关测试。
重点检查数据泄漏、模板与 loss mask、reward 正确性、实验可复现性。
先输出问题、文件行号、影响与最小测试，不要直接替我修复。
你实际运行了哪些测试就写哪些；没有运行的明确标注 NOT RUN。
审查之后再用一道临时变式题检查我是否真正理解代码。
```

## 6.5 确认本章完成

```text
请按 CXX 验收标准检查我的解释、测试、实验日志和报告。
分别给出：学习掌握状态、工程执行程度、效果证据。
满足要求才更新为 PASSED；受算力阻塞写 BLOCKED；没有训练就不要说训练完成。
列出下一章唯一的启动任务，不要同时展开后续所有章节。
```

## 6.6 你的第一个里程碑

不是“学完 GRPO”，也不是“生成 5000 条数据”。而是：

> **你能够解释一条工具任务如何变成 token 与 loss；本地有可复现的工具环境、独立验证器和一个真实评测过的 Tool-use SFT 基线。**

先做到这里，再进入奖励优化，会更容易判断后面的问题究竟来自数据、模型、RL 还是工具环境。

---

## 资料链接定义

[R01]: https://huggingface.co/learn/smol-course/en/unit0/1
[R02]: https://cs336.stanford.edu/
[R03]: https://huggingface.co/docs/trl/en/dataset_formats
[R04]: https://huggingface.co/docs/trl/en/sft_trainer
[R05]: https://huggingface.co/docs/transformers/en/chat_templating
[R06]: https://huggingface.co/docs/trl/en/chat_templates
[R07]: https://huggingface.co/docs/transformers/en/chat_extras
[R08]: https://huggingface.co/docs/peft/en/developer_guides/quantization
[R09]: https://huggingface.co/Qwen/Qwen3-0.6B
[R10]: https://huggingface.co/Qwen/Qwen3-1.7B
[R11]: https://huggingface.co/docs/trl/en/reducing_memory_usage
[R12]: https://huggingface.co/docs/trl/en/dpo_trainer
[R13]: https://github.com/openai/spinningup/blob/master/docs/spinningup/rl_intro3.rst
[R14]: https://arxiv.org/abs/2212.10560
[R15]: https://arxiv.org/abs/2305.18290
[R16]: https://arxiv.org/abs/2402.03300
[R17]: https://huggingface.co/docs/trl/en/grpo_trainer
[R18]: https://huggingface.co/docs/trl/en/distillation_trainer
[R19]: https://arxiv.org/abs/2306.13649
[R20]: https://verl.readthedocs.io/en/latest/start/agentic_rl.html
[R21]: https://verl.readthedocs.io/en/latest/advance/agent_loop.html
[R22]: https://pytorch.org/get-started/locally/
[R23]: https://developer.nvidia.com/cuda/gpus
[R24]: https://pytorch.org/blog/pytorch-2-7/
[R25]: https://developers.openai.com/codex/guides/agents-md/
[R26]: https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
[R27]: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
[R28]: https://github.com/openai/spinningup/blob/master/docs/algorithms/ppo.rst
[R29]: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
[V01]: https://www.bilibili.com/video/BV1KFpczQEkA/
[V02]: https://www.bilibili.com/video/BV18hsFzbEKJ/
[V03]: https://www.bilibili.com/video/BV1N16ZBuERA/
[V04]: https://www.bilibili.com/video/BV1MizSBJEbi/
[V05]: https://www.bilibili.com/video/BV1d4Yvz4EXA/
[VSEARCH]: https://search.bilibili.com/all?keyword=%E4%BA%94%E9%81%93%E5%8F%A3%E7%BA%B3%E4%BB%80%20MultiTurn%20Tool%20Use
