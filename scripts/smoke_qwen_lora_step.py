#!/usr/bin/env python
"""Run exactly one offline LoRA optimizer step on the fixed Qwen3.5 sample."""

from __future__ import annotations

import gc
import hashlib
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoProcessor
from trl.chat_template_utils import get_training_chat_template

from inspect_tokens import MESSAGES, build_labels, validate_labels


SEED = 20260915
MAX_LENGTH = 128
LEARNING_RATE = 1e-4
LORA_RANK = 4
LORA_ALPHA = 8
OUTPUT_DIR = Path("runs/C00-QWEN-LORA-STEP-001/adapter")
TARGET_MODULES = [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def adapter_checksum(model: torch.nn.Module) -> str:
    """Hash PEFT weights in a device- and serialization-independent order."""
    digest = hashlib.sha256()
    state = get_peft_model_state_dict(model)
    for name in sorted(state):
        tensor = state[name].detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_base_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model.to(device)


def main() -> None:
    started_at = time.perf_counter()
    model_path = os.environ.get("QWEN35_08B_PATH")
    if not model_path:
        raise RuntimeError("请先在仓库根目录执行：source scripts/env.sh")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录：{OUTPUT_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 BF16")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    training_template = get_training_chat_template(processor) or processor.chat_template
    encoded = processor.apply_chat_template(
        MESSAGES,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        chat_template=training_template,
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"]
    assistant_mask = encoded["assistant_masks"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        assistant_mask = assistant_mask[0]
    if len(input_ids) > MAX_LENGTH:
        raise RuntimeError(f"样本长度 {len(input_ids)} 超过上限 {MAX_LENGTH}，拒绝截断")

    labels = build_labels(input_ids, assistant_mask)
    validate_labels(input_ids, assistant_mask, labels)
    batch_input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    batch_attention_mask = torch.ones_like(batch_input_ids)
    batch_labels = torch.tensor([labels], dtype=torch.long, device=device)

    model = load_base_model(model_path, device)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.train()

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable:
        raise RuntimeError("没有可训练参数")
    if any("visual" in name for name in trainable):
        raise RuntimeError("LoRA 意外匹配到视觉模块")
    trainable_tensor_count = len(trainable)
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable.values())
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}

    optimizer = torch.optim.AdamW(trainable.values(), lr=LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        labels=batch_labels,
    )
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"loss 非有限值：{loss.item()}")
    loss_value = loss.detach().float().item()
    loss.backward()

    gradients = [parameter.grad for parameter in trainable.values() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("所有可训练参数的梯度均为空")
    if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("梯度包含 NaN 或 Inf")
    nonzero_gradient_tensors = sum(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    if nonzero_gradient_tensors == 0:
        raise RuntimeError("所有 LoRA 梯度均为 0")

    optimizer.step()  # The single authorized optimizer step.
    changed_trainable_tensors = sum(
        not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in trainable.items()
    )
    if changed_trainable_tensors == 0:
        raise RuntimeError("optimizer step 后没有 LoRA 张量发生变化")

    trained_checksum = adapter_checksum(model)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 1024**2

    del outputs, loss, optimizer, before, trainable, gradients, model
    gc.collect()
    torch.cuda.empty_cache()

    reload_base = load_base_model(model_path, torch.device("cpu"))
    reloaded = PeftModel.from_pretrained(reload_base, OUTPUT_DIR, is_trainable=False)
    reloaded_checksum = adapter_checksum(reloaded)
    if reloaded_checksum != trained_checksum:
        raise RuntimeError("保存前后 adapter checksum 不一致")

    elapsed_seconds = time.perf_counter() - started_at
    print(f"model_path={model_path}")
    print(f"model_class={type(reload_base).__name__}")
    print("dtype=torch.bfloat16")
    print("attention_implementation=sdpa")
    print(f"sequence_length={len(input_ids)}")
    print(f"effective_labels={sum(label != -100 for label in labels)}")
    print(f"lora_rank={LORA_RANK}")
    print(f"lora_alpha={LORA_ALPHA}")
    print(f"trainable_tensors={trainable_tensor_count}")
    print(f"trainable_parameters={trainable_parameter_count}")
    print(f"loss={loss_value}")
    print(f"nonzero_gradient_tensors={nonzero_gradient_tensors}")
    print(f"changed_trainable_tensors={changed_trainable_tensors}")
    print(f"adapter_checksum={trained_checksum}")
    print("adapter_reload=PASS")
    print(f"peak_allocated_mib={peak_allocated_mib:.2f}")
    print(f"peak_reserved_mib={peak_reserved_mib:.2f}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
