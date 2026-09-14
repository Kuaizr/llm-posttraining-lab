#!/usr/bin/env python
"""Inspect one local Qwen3.5 tool-SFT sample without loading model weights."""

from __future__ import annotations

import os

from transformers import AutoProcessor
from trl.chat_template_utils import get_training_chat_template


IGNORE_INDEX = -100


MESSAGES = [
    {"role": "system", "content": "你是办公助手"},
    {"role": "user", "content": "查询订单 17 的金额"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_order", "arguments": {"id": 17}},
            }
        ],
    },
    {
        "role": "tool",
        "name": "get_order",
        "content": '{"id":17,"amount":120}',
    },
    {"role": "assistant", "content": "订单 17 的金额是 120 元"},
]


def build_labels(input_ids: list[int], assistant_mask: list[int]) -> list[int]:
    """Copy supervised token IDs and mask every other target with IGNORE_INDEX."""
    if len(input_ids) != len(assistant_mask):
        raise ValueError("input_ids 与 assistant_mask 长度不一致")

    labels = [IGNORE_INDEX for _ in range(len(input_ids))]
    for i, mask_bit in enumerate(assistant_mask):
        if mask_bit not in (0, 1):
            raise ValueError(f"assistant_mask[{i}] 不是 0 或 1：{mask_bit!r}")
        if mask_bit == 1:
            labels[i] = input_ids[i]
    return labels


def validate_labels(
    input_ids: list[int], assistant_mask: list[int], labels: list[int]
) -> None:
    """Fail early if the constructed supervision region is inconsistent."""
    assert len(input_ids) == len(assistant_mask) == len(labels), "三者长度不一致"
    assert set(assistant_mask) == {0, 1}, "本检查样本必须同时包含 mask 0 和 1"
    assert any(label != IGNORE_INDEX for label in labels), "所有 label 都被 mask"

    for index, (token_id, mask_bit, label) in enumerate(
        zip(input_ids, assistant_mask, labels, strict=True)
    ):
        expected_label = token_id if mask_bit == 1 else IGNORE_INDEX
        assert label == expected_label, (
            f"位置 {index} 的 label 错误：期望 {expected_label}，实际 {label}"
        )


def main() -> None:
    model_path = os.environ.get("QWEN35_08B_PATH")
    if not model_path:
        raise RuntimeError("请先在仓库根目录执行：source scripts/env.sh")

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    training_template = get_training_chat_template(processor)
    if training_template is None:
        training_template = processor.chat_template

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

    # Some processors return a batch dimension and some return a flat sequence.
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        assistant_mask = assistant_mask[0]

    labels = build_labels(input_ids, assistant_mask)
    validate_labels(input_ids, assistant_mask, labels)

    print("index\ttoken\tassistant_mask\tlabel")
    for index, (token_id, mask_bit, label) in enumerate(
        zip(input_ids, assistant_mask, labels, strict=True)
    ):
        token = processor.tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"{index}\t{token!r}\t{mask_bit}\t{label}")


if __name__ == "__main__":
    main()
