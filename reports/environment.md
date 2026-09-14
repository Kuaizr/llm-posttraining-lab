# C00 环境体检与安装方案

> 日期：2026-09-14  
> 当前状态：PyTorch/Hugging Face 栈、随机微型 SFT dry run、三个固定模型 snapshot 和真实模型 GPU 推理 smoke 已完成；真实模型 backward、optimizer step 和正式训练均未执行。

## 1. 已确认事实

| 项目 | 实际值 | 证据命令 |
|---|---|---|
| 正式项目目录 | `/mnt/d/llm-posttraining-lab` | `pwd` |
| 运行方式 | Arch Linux on WSL2，Linux 6.6.114.1 | `uname -a`、读取 `/etc/os-release` |
| glibc | 2.43 | `ldd --version` |
| uv | 0.11.19 | `uv --version` |
| 系统 Python | `/usr/sbin/python`，3.14.5，无 pip | `command -v python`、`python --version`、`python -m pip --version` |
| 已有 uv Python | CPython 3.12.13，位于 `/home/kzer/.local/share/uv/python/`，约 109 MB | `uv python list --only-installed`、`uv python dir`、`du -shL` |
| 当前 uv cache | `/home/kzer/.cache/uv`；后续不得继续用于本项目 | `uv cache dir` |
| GPU | NVIDIA GeForce RTX 5070 Ti，16303 MiB；检查时可用 14830 MiB | `nvidia-smi --query-gpu=...` |
| 驱动/UMD | Windows KMD 610.47；CUDA UMD 13.3 | `nvidia-smi` |
| nvcc | 未安装/不可见 | `command -v nvcc`、`nvcc --version` |
| 当前训练包 | torch 2.14.0+cu130、torchvision 0.29.0+cu130、Transformers 5.17.0、Datasets 5.0.1、Accelerate 1.15.0、PEFT 0.20.0、TRL 1.13.0；未安装 bitsandbytes | `uv pip list --python .venv/bin/python`、导入 smoke |
| D 盘容量 | 总计约 753 GB，第二阶段完成后剩余约 591 GB | `df -h /mnt/d`，2026-09-14 13:42 CST |
| 模型缓存 | 三个固定 Qwen3.5 snapshot 位于 `/mnt/d/llm-posttraining-cache/huggingface/hub` | `hf cache verify` |

`nvidia-smi` 成功只证明 WSL2 能看到 GPU 和驱动。没有 `nvcc` 不代表预编译 PyTorch wheel 无法使用；PyTorch CUDA 是否可用仍需安装后实际执行 kernel、backward 和 synchronize。

## 2. 官方兼容信息

- NVIDIA 官方表将 GeForce RTX 5070 Ti 列为 compute capability 12.0：[CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)。
- PyTorch 2.14 于 2026-09-02 发布；官方矩阵列出 CUDA 12.6、13.0、13.2，默认 wheel 为 CUDA 13.0，Python 支持 3.10～3.15：[PyTorch 2.14 Release Blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)。
- PyTorch 官方建议用预编译 binary，并用 `torch.cuda.is_available()` 验证驱动可访问性：[Start Locally](https://pytorch.org/get-started/locally/)。
- uv 支持把 cache 改到指定目录，并建议 cache 与虚拟环境位于同一文件系统：[uv cache](https://docs.astral.sh/uv/concepts/cache/)、[uv environments](https://docs.astral.sh/uv/pip/environments/)。

据此推断：驱动报告的 CUDA UMD 13.3 能覆盖 CUDA 13.0 wheel 所需的驱动能力，PyTorch 2.14.0 + cu130 是合理的第一候选；最终兼容性必须由本机 smoke test 证明，不能只凭版本号宣称成功。

## 3. 建议方案

选择已有 CPython 3.12.13，不使用系统 Python 3.14.5。Python 3.12并非 PyTorch 的硬性要求，而是为了降低 Transformers/TRL/PEFT 等周边依赖的兼容变量。

所有新增的大体积内容放在 D 盘：

```text
/mnt/d/llm-posttraining-lab/.venv
/mnt/d/llm-posttraining-cache/uv
/mnt/d/llm-posttraining-cache/huggingface
/mnt/d/llm-posttraining-cache/torch
/mnt/d/llm-posttraining-cache/tmp
```

第一阶段只创建 `.venv` 并安装 `torch==2.14.0` 的 CUDA 13.0 wheel。不安装 torchvision、torchaudio、Transformers、TRL、PEFT、bitsandbytes、FlashAttention、vLLM 或 verl。

实际执行步骤：

```bash
export UV_CACHE_DIR=/mnt/d/llm-posttraining-cache/uv
export HF_HOME=/mnt/d/llm-posttraining-cache/huggingface
export TORCH_HOME=/mnt/d/llm-posttraining-cache/torch
export TMPDIR=/mnt/d/llm-posttraining-cache/tmp

uv venv --python /home/kzer/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 .venv
uv pip install --python .venv/bin/python torch==2.14.0 --index https://download.pytorch.org/whl/cu130
```

安装后依次验证：

1. 记录 Python、torch、`torch.version.cuda`、GPU 名称、compute capability 和 `torch.cuda.get_arch_list()`。
2. `torch.cuda.is_available()` 必须为 True。
3. 在 GPU 上执行小型 FP32 与 BF16 tensor 运算。
4. 执行一个小矩阵乘法、标量 loss、backward 和 `torch.cuda.synchronize()`。
5. 检查结果及梯度均为 finite；任何 kernel/架构错误立即停止，不继续安装 Hugging Face 栈。

第二阶段只有在第一阶段通过后，才另行确认并安装 Transformers、Datasets、PEFT、TRL、Accelerate 和测试工具。模型下载单独授权，第一、二阶段都不下载模型。

## 4. 第一阶段实际结果

第一阶段于 2026-09-14 执行成功：

```text
python=3.12.13
torch=2.14.0+cu130
torch_cuda_runtime=13.0
cuda_available=True
device_name=NVIDIA GeForce RTX 5070 Ti
compute_capability=(12, 0)
compiled_arch_list=['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
FP32 forward/backward: finite
BF16 forward/backward: finite
peak_memory_allocated_mib=65.38
smoke_test=PASS
```

测试使用 256×256 CUDA tensor，分别执行 FP32 与 BF16 矩阵乘法、标量 loss、backward 和 `torch.cuda.synchronize()`。没有模型训练。

安装后 `.venv` apparent size 约 5.2 GB；D 盘剩余约 600 GB，实际新增量在 15 GB 上限内。环境精确解析结果见 `requirements/C00-torch-cu130.lock`。

导入 torch 时出现 `No module named 'numpy'` 警告，因为第一阶段严格只安装 torch；不影响本次纯 torch CUDA tensor 测试。NumPy 未擅自补装，将随第二阶段 Hugging Face 栈一并解析。

## 5. 第一阶段预算与影响

| 项目 | 上限/范围 |
|---|---|
| 付费 API | 0；不调用 |
| 模型下载 | 0；不下载 |
| 网络来源 | `download.pytorch.org` 及解析 wheel 依赖所需的 Python 包索引 |
| 下载量 | 精确值在解析前未知；按不超过 6 GB 控制，超过则停止并报告 |
| D 盘新增占用 | 按不超过 15 GB 控制，包括 uv cache、wheel 解包和 `.venv` |
| 最长安装时间 | 30 分钟；超时停止 |
| GPU 操作 | 仅数秒的小 tensor forward/backward smoke test，预计显存远低于 1 GB；不是模型训练 |
| 修改范围 | 只创建上述 D 盘 cache 目录和项目 `.venv`；不改系统 Python、驱动、CUDA Toolkit 或 C 盘仓库 |

## 6. 当前结论

- 操作系统、驱动可见性、GPU 型号、磁盘与候选软件组合：已核验。
- PyTorch CUDA/kernel/backward：`SMOKE_ONLY`，结果 PASS；FP32/BF16 输出和梯度均 finite。
- 随机微型 causal LM：CPU forward/backward、LoRA 保存重载、TRL SFT 1 个 optimizer step 均 PASS。
- 真实 Qwen3.5：0.8B Base GPU forward、0.8B/2B 后训练版短生成、0.8B 未训练 LoRA 注入/保存/重载 PASS；真实模型 backward/optimizer step 仍为 `NOT RUN`。
- 当前证据支持进入真实小模型 SFT 的下一步准备，但不支持宣称训练收敛、2B PPO 可运行或已获得效果提升。

## 7. 第二阶段实际结果

### 7.1 Hugging Face 栈

在已有 D 盘 `.venv` 中通过 `uv` 安装稳定发布版；解析未替换 torch，只将 `fsspec` 从 2026.7.0 调整为 Datasets 兼容的 2026.6.0。随后按 Qwen3.5 官方模型卡补装匹配 cu130 的 torchvision 和 Pillow。

```text
python=3.12.13
torch=2.14.0+cu130
torchvision=0.29.0+cu130
transformers=5.17.0
datasets=5.0.1
accelerate=1.15.0
peft=0.20.0
trl=1.13.0
numpy=2.5.3
pillow=12.3.0
pytest=9.1.1
```

完整解析见 `requirements/C00-hf-stack.lock`。`.venv` apparent size 约 5.7GB。

### 7.2 无下载 CPU smoke

完全离线、随机初始化的两层 GPT-2 配置完成：

- 内存 Dataset 构造；
- causal LM loss 和 backward，loss finite；
- LoRA 可训练参数存在；
- adapter 保存为 `adapter_model.safetensors` 并成功重载；
- `SFTConfig`、`DPOConfig`、`GRPOConfig` 导入。

随后通过 `SFTTrainer` 对两条内存文本执行 1 个 CPU optimizer step：

```text
training_loss=2.274243116378784
global_step=1
changed_trainable_tensors=2
adapter_saved=True
```

该结果只证明训练软件链路和一次参数更新可运行，不是模型效果实验。

### 7.3 固定模型 snapshot

| 模型 | revision | cache verify | apparent size |
|---|---|---:|---:|
| `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 12 files PASS | 1.7GB |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 13 files PASS | 1.7GB |
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 13 files PASS | 4.3GB |

下载中 Qwen3.5-2B 的 Xet 路径失速。核对进程后终止单个旧下载 PID，改用普通 HTTP 完成；完整 snapshot 校验通过后，删除了旧 Xet 留下且未被 snapshot 引用的一个 469MB `.incomplete` 缓存片段。

### 7.4 真实模型 GPU smoke

所有加载均使用固定本地 snapshot、BF16、`local_files_only=True`，没有训练：

| 检查 | 结果 | 峰值 allocated | 峰值 reserved |
|---|---|---:|---:|
| Qwen3.5-0.8B Base，3-token forward | logits finite，shape `[1, 3, 248320]` | 1687.18MiB | 1704.0MiB |
| Qwen3.5-0.8B Base，3-token causal loss forward | loss 10.39895，finite；`backward=False` | 1687.18MiB | 1704.0MiB |
| Qwen3.5-0.8B，短 greedy generation | 输出 `OK` | 1687.68MiB | 1704.0MiB |
| Qwen3.5-2B，短 greedy generation | 输出 `OK` | 4282.27MiB | 4288.0MiB |

本地 `processor.apply_chat_template` 使用 `enable_thinking=False`；OpenAI-compatible API 示例中的嵌套 `chat_template_kwargs` 不能原样传给 processor。非 thinking 模板仍会插入一个空的 `<think>...</think>` 区段，这是当前官方模板的预期结构。

Transformers 报告 `causal_conv1d` 与 `flash-linear-attention` 未安装，并正确回退到较慢的 PyTorch 实现。本阶段按项目规则不自动安装这些优化扩展；这不是正确性失败，但后续训练时间需要实测。

## 8. 快速进入环境

从正式仓库根目录执行：

```bash
source scripts/env.sh
python -c "import torch, transformers, datasets, accelerate, peft, trl; print(torch.__version__, transformers.__version__, trl.__version__)"
```

`scripts/env.sh` 只设置本项目的 D 盘缓存路径并激活现有 `.venv`，不下载模型、不启动训练。

脚本还导出固定 snapshot 路径：`QWEN35_08B_BASE_PATH`、`QWEN35_08B_PATH`、`QWEN35_2B_PATH`，避免学习代码依赖浮动 `main` 或再次联网解析 revision。
