# C00 环境体检与安装方案

> 日期：2026-09-14  
> 当前状态：第一阶段环境创建与 PyTorch CUDA smoke test 已完成；模型下载与 GPU 训练均未执行。

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
| 当前训练包 | torch、transformers、datasets、peft、trl、accelerate、bitsandbytes 均未安装 | Python `importlib.metadata` 查询 |
| D 盘容量 | 总计约 753 GB，剩余约 605 GB | `df -h /mnt/d` |
| 模型缓存 | 未发现 Hugging Face cache | 检查 `/home/kzer/.cache/huggingface` |

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
- 模型 forward/backward、LoRA 保存重载：`NOT RUN`。
- 当前已有最小 PyTorch CUDA 能力证据，但没有证据宣称 Transformers/TRL、模型或 LoRA 训练链路可用。
