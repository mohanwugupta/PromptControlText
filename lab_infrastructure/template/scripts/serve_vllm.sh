#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to a Hugging Face model ID or local model path}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL")}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

mkdir -p "${SPAR_SCRATCH:-/root/scratch}"
export HF_HOME="${HF_HOME:-${SPAR_SCRATCH:-/root/scratch}/hf}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${SPAR_SCRATCH:-/root/scratch}/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SPAR_SCRATCH:-/root/scratch}/triton}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_DIR" "$TRITON_CACHE_DIR"

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --dtype auto
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-chunked-prefill
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=($EXTRA_VLLM_ARGS)
  ARGS+=("${EXTRA[@]}")
fi

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
