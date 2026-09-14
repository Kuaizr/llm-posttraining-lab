# 学习会话：2026-09-14 / C00 / 后训练数据—目标—更新—评测地图

> 本文由 Codex 在学习者明确请求后生成，帮助等级为“实现示范”。它是学习材料，不是学习者独立完成的掌握证据；需通过新的变式题独立验收。

## 本次唯一目标

建立 SFT、DPO、GRPO 的数据—反馈—更新—评测对照，并把 RLHF、RLVR、蒸馏和 Agentic RL 放到同一张地图中。

本次不创建 Python 环境，不安装依赖，不下载模型，不运行 GPU 训练，也不进入 C01。

## 诊断与上次复习

- 诊断问题：可靠 verifier 下，同一 prompt 的 8 条 rollout 全部失败且奖励均为 0，直接做 GRPO 是否有有效组内学习信号？
- 学习者原始回答：不知道。
- 需要补学：同组奖励全同会使相对 advantage 为 0；可靠 verifier 不等于当前策略具有可学习的探索信号。
- 使用的帮助等级：明确答案 + 实现示范。

## 本次阅读（最多两项）

| 资料链接 | 指定章节/范围 | 对应机制 | 本项目中的用途 |
|---|---|---|---|
| [Smol Course：Instruction Tuning](https://huggingface.co/learn/smol-course/en/unit1/1) | What is Instruction Tuning；Module Overview；只浏览 Preference Alignment 与 Model Evaluation 的概览 | instruction tuning、SFT、chat template、训练后评测 | 恢复从预训练 checkpoint 到指令模型的基本路径 |
| [TRL：Dataset formats and types](https://huggingface.co/docs/trl/en/dataset_formats) | Overview；Language modeling；Prompt-only；Prompt-completion；Preference；Tool Calling；trainer 对照表 | 区分数据格式与任务类型，以及 SFT/DPO/GRPO 的输入契约 | 设计内部轨迹到各 trainer 输入的导出层 |

## 核心对照

| 方法 | 训练输入 | 监督或反馈信号 | 当前策略在线采样 | 更新对象 | 适合解决的问题 | 常见误用 | 独立评测 |
|---|---|---|---|---|---|---|---|
| SFT | 完整语言建模文本，或 `prompt + completion`；对话数据通常是 `messages`，工具 SFT 还需工具 schema 和真实 observation | 数据中给定的目标 token；用 teacher forcing 计算 next-token cross-entropy。可用 label mask 只训练 assistant/目标区域 | 否。训练样本在优化前已经存在 | 学生策略参数；可全参数更新或只更新 LoRA 等适配器 | 教会稳定格式、工具 schema、基本调用顺序；在策略几乎无法成功探索时先建立行为先验 | 把伪造的 tool result 当真实 observation；把 oracle/测试答案放进 prompt；mask、截断或 EOS 处理错误 | 封存任务上的 task success、invalid tool call rate、no-tool correctness；同时检查 loss mask 与保存重载 |
| DPO | 同一 prompt/共同前缀下的 `chosen` 和 `rejected` completion；偏好对通常预先构造并保存 | 成对偏好。优化策略对 chosen/rejected 的序列 log-prob 差距，并与冻结 reference policy 的对应差距比较 | 标准离线 DPO 训练时不需要；偏好数据可在训练前由其他流程采集 | 当前策略参数；reference policy 冻结 | 已有可信相对偏好，但难以给每个答案定义精确标量 reward 的场景 | chosen/rejected 的前缀或可见 observation 不同；偏好只是长度、格式等伪信号；把环境 observation 当模型动作优化 | 独立成对偏好准确率、task success、长度分布与关键失败类型；不能只看 DPO loss |
| GRPO | 通常是 prompt-only 加任务/环境元数据；completion/trajectory 在训练时生成 | verifier/reward 对在线样本打分，再在同一 prompt 的 rollout 组内形成相对 advantage；策略目标用 advantage 加权生成 token 的 log-prob，具体实现还可能含 ratio clipping 与参考策略 KL | 是。必须由当前策略或明确的行为策略产生一组 rollout | 当前策略参数；计算 ratio 时需区分 old/behavior policy 与 current policy，reference policy 若使用则冻结 | 有可靠可验证反馈、当前策略能偶尔成功、同组奖励有差异且 rollout 成本可控时，提高任务成功概率 | 奖励全同或全零；混合不同 prompt 做组内归一化；奖励可被投机；采样 token、log-prob 与 mask 没对齐 | 封存 IID/OOD task success、invalid tool call rate、工具/token/时间成本、失败类型；训练 reward 只作诊断，不代替独立评测 |

### 三种方法的数据流

```text
SFT
固定 prompt + 专家 completion
        → teacher forcing / token labels
        → 更新策略

DPO
固定 prompt + chosen + rejected
        → 当前策略与冻结 reference 的成对 log-prob 对比
        → 更新当前策略

GRPO
固定 prompt + 环境元数据
        → 当前策略在线采样一组 completion/trajectory
        → verifier 给 reward
        → 同一 prompt 组内 reward 变成 advantage
        → 用 advantage 加权策略更新
```

## 其他概念放在什么位置

| 概念 | 数据/反馈 | 与核心三种方法的关系 |
|---|---|---|
| RLHF | 人类偏好或由人类偏好训练出的 reward model | 是一类反馈来源与训练流水线，不等同于某个单一优化器；可结合 PPO 等在线 RL，也可产生 DPO 所需偏好对 |
| RLVR | 可程序验证的 reward，例如数学答案、代码测试或本项目 oracle | 省去主观 reward model 的一部分需求，但 verifier 的正确性、覆盖率和防投机仍是核心风险；GRPO 常用于这类场景 |
| 蒸馏 | 教师输出、概率分布、排序或在线教师反馈，取决于接口能力 | 可用 SFT 模仿教师结果，也可有其他蒸馏目标；闭源 API 只返回文本时，不能假装获得完整 token 分布 |
| Agentic RL | 模型动作与环境 observation 交替形成多轮 trajectory，再依据轨迹结果更新策略 | “能调用工具”不等于 Agentic RL；必须有学生策略 rollout、环境执行、reward/advantage 和真实参数更新 |

## 三个项目问题

### 没有数据时先做什么

先定义任务，而不是先选训练器：固定工具 schema、环境状态、成功/失败条件、调用预算和独立 oracle/verifier；随后设计 train/dev/test-IID/test-OOD 的任务族切分。先用 mock 或手工构造少量虚构任务验证环境与评测器，再生产训练轨迹。没有可靠任务与评测时，SFT、DPO 和 RL 都无法判断是否真的改善。

### 为什么工具 SFT 通常先于复杂 RL

若初始策略几乎不会生成合法工具名和参数，在线 rollout 可能全部失败，同组奖励没有差异，RL 得不到有效方向。经过验证的工具 SFT 可以先建立格式、schema 和基本调用顺序，使策略进入“偶尔成功、失败可区分”的区域。它不是 RL 的理论强制前置，但通常能降低探索成本和无效采样比例。

### RL 为什么仍然需要数据

RL 不需要为每个 prompt 预存唯一 gold completion，但仍需要：任务 prompt、环境状态、工具定义、预算与终止规则、可验证 reward，以及独立评测集。训练过程中产生的 rollout 本身也是数据，必须记录策略版本、采样配置、工具执行结果、奖励和失败原因。没有这些数据契约，reward 无法复现，参数更新也无法审计。

## 两个重要边界

### 全同奖励

若同一 prompt 的奖励是 `[0, 0, 0, 0]`，以组内均值为基线时：

```text
mean = 0
advantage = [0, 0, 0, 0]
```

此时没有奖励驱动的相对策略梯度。应先检查 verifier，再考虑经过验证的 SFT bootstrap、降低任务难度、课程学习或改善探索；不能通过篡改 oracle 或把错误轨迹标成成功来制造奖励差异。

### Base checkpoint 与学习起点

真正的 pretrained base checkpoint 主要经过 next-token 预训练；instruction-tuned/chat checkpoint 已经经过某种后训练。即使后者被当作本实验的“起始模型”或“baseline”，也不能把它称为未经后训练的 Base。报告必须记录完整模型 ID、revision 和 chat template 版本。

## 本次实现与运行

- 修改文件：`notes/C00_map.md`。
- 核心内容由谁完成：Codex 实现示范，不是学习者独立完成。
- 实际执行：仅写入 Markdown 教学记录。
- 未运行：依赖安装、CUDA tensor、模型前向/反向、LoRA 保存重载和训练，均为 `NOT RUN`。

## 尚未解决的问题

- 学习掌握：需要用新情境独立区分固定偏好数据与在线 rollout，并定义训练外评测指标。
- 工程环境：尚未用 `uv` 创建独立 Python 环境；PyTorch CUDA、compute capability、forward/backward 与保存重载均未实测。
- 实验效果：`NOT_EVALUATED`。

## 本次结束

- 目标：部分完成。知识地图已生成，但不是学习者独立完成，且 C00 环境验收尚未完成。
- 证据路径：`notes/C00_map.md`、`reviews/2026-09-14_C00_demo.md`、`PROGRESS.md`。
- 需复习的一个知识点：可靠 verifier 与有效探索/非全同 reward 是两个不同条件。
- 下一次唯一任务：学习者不看本文表格，独立完成一题新的 SFT/DPO/GRPO 路径选择题。
- `PROGRESS.md` 是否已按事实更新：是。

