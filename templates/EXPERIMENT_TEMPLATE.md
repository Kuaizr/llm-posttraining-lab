# 实验报告：<experiment_id>

> 复制到 `reports/` 后填写。未知数字留空或写 UNKNOWN；未运行写 NOT RUN。不要填入计划中的预期结果。

## 1. 问题与假设

- 所属章节：
- 本次只改变什么：
- 对照组与父 checkpoint：
- 假设与可能的反证：
- 不变的评测任务/预算：

## 2. 数据与反馈契约

- 数据/任务生成器版本、随机种子、hash：
- train / dev / test-IID / test-OOD 数量：
- family/instance 隔离方式与去重检查：
- 本次是否接触封存 test：否 / 是（用途及是否已降级为 dev）。
- 教师模型/版本/提示词版本/来源：
- 实际工具输出的保存位置与重放测试：
- 数据过滤数量与拒绝原因：
- verifier/reward 版本、边界测试与反例：
- gold/oracle 是否可能进入模型输入：

## 3. 模型、环境与配置

| 项目 | 实际配置 |
|---|---|
| OS / GPU / 驱动 / Python | |
| PyTorch / CUDA / Transformers / TRL / PEFT | |
| 模型 ID / revision / 父 checkpoint | |
| tokenizer / 模板版本 / thinking 设置 | |
| LoRA / 量化 / dtype | |
| 序列长度 / 生成长度 / micro-batch / 梯度累积 | |
| 优化器 / 学习率 / seed / optimizer steps | |
| GRPO group size / 采样配置 / old/ref 来源 | |
| 最大工具调用 / 回合数 / 超时 | |
| 训练停止条件与资源授权 | |

非本实验适用的项写“不适用”，不要猜值。区分 generation 调用次数、microsteps 和 optimizer updates。

## 4. 实际执行

```text
<真正执行的命令；未执行的单列，不能混在这里>
```

- 代码 commit 或快照/hash：
- 配置文件路径：
- 日志/轨迹/checkpoint 路径：
- 保存→重载验证：
- 实际执行测试与输出：
- 未运行项及原因：
- 执行程度：NOT_RUN / SMOKE_ONLY / FULL_RUN。

## 5. 资源与成本

- 实测峰值显存与测量方式：
- 实测运行时长/吞吐：
- 教师请求数、输入/输出 tokens、重试和缓存命中：
- 已知实际 API 费用：
- 价格未知部分：UNKNOWN，不能按 0 元计。
- 是否达到预算/停止上限：

## 6. 真实评测结果

| 指标 | 对照 | 实验 | 分母/样本数及定义 |
|---|---|---|---|
| 任务成功率 | | | 包含超时、格式错与工具错误任务 |
| 无效工具调用率 | | | |
| 错误恢复成功率 | | | |
| 平均工具调用数 | | | 全体任务；成功任务可另报 |
| 平均生成 tokens / 截断率 | | | |
| 独立 OOD 成功率 | | | 最终封存评测前标记未评测 |

- 本次评测用 dev 还是 test：
- 重复运行/随机种子：
- 不确定性、样本量限制：
- 训练 reward 与外部成功率是否一致：
- 不能据此得出的结论：

## 7. 失败分析

| task_id | 失败发生位置 | 证据 | 判断原因 | 下一步最小验证 |
|---|---|---|---|---|

失败样本不从分母剔除。工具、环境、模型、奖励问题分开记录。

## 8. 结论

- 效果证据：NOT_EVALUATED / NO_GAIN / INCONCLUSIVE / GAIN_WITH_EVIDENCE。
- 学到了什么：
- 当前主要瓶颈与证据：
- 下一次只改变什么：
- 是否需要新的授权/算力：
