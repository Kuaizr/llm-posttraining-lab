#!/usr/bin/env bash
# Codex-provided environment bootstrap for this specific WSL2/D-drive lab.
# Source this file from the repository root: source scripts/env.sh

export UV_CACHE_DIR=/mnt/d/llm-posttraining-cache/uv
export HF_HOME=/mnt/d/llm-posttraining-cache/huggingface
export HF_HUB_CACHE=/mnt/d/llm-posttraining-cache/huggingface/hub
export TORCH_HOME=/mnt/d/llm-posttraining-cache/torch
export TMPDIR=/mnt/d/llm-posttraining-cache/tmp

export QWEN35_08B_BASE_PATH=/mnt/d/llm-posttraining-cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
export QWEN35_08B_PATH=/mnt/d/llm-posttraining-cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
export QWEN35_2B_PATH=/mnt/d/llm-posttraining-cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc

if [[ -f /mnt/d/llm-posttraining-lab/.venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /mnt/d/llm-posttraining-lab/.venv/bin/activate
else
    echo "Missing project environment: /mnt/d/llm-posttraining-lab/.venv" >&2
    return 1 2>/dev/null || exit 1
fi
