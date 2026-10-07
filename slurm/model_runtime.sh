#!/bin/bash
# Shared serving defaults for the smoke test and full generation launcher.
# DeepSeek's workers crashed with CUDA illegal memory access during inference.
# Eager, synchronous execution is a mitigation and helps isolate kernel faults.
VLLM_MODEL_ARGS=()
if [[ "$MODEL_SLUG" == deepseek_r1_distill_* ]]; then
    : "${VLLM_MAX_NUM_SEQS:=32}"
    : "${VLLM_MAX_NUM_BATCHED_TOKENS:=4096}"
    : "${GENERATION_MAX_WORKERS:=16}"
    VLLM_MODEL_ARGS+=(--enforce-eager --no-async-scheduling)
else
    : "${VLLM_MAX_NUM_SEQS:=512}"
    : "${VLLM_MAX_NUM_BATCHED_TOKENS:=32768}"
    : "${GENERATION_MAX_WORKERS:=64}"
fi
