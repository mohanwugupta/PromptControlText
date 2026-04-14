#!/bin/bash
#SBATCH --job-name=smoke_test_prompt_control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# =============================================================================
# Prompt-Level Safety Controllers SMOKE TEST
# =============================================================================

set -eo pipefail

echo "=========================================="
echo "Running SMOKE TEST (Qwen2.5-72B limited to 5 prompts)"
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"

# ------------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/PromptControlText
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct
SERVED_MODEL_NAME=$(basename "$MODEL_PATH")
CONDA_ENV=PromptControlText
VLLM_PORT=8000
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.92

# ------------------------------------------------------------------
# 1. Environment setup
# ------------------------------------------------------------------
cd "$PROJECT_DIR"
module load anaconda3/2025.6

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
else
    source activate "$CONDA_ENV"
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------------
# 2. Pre-flight: scorer unit tests (no GPU required — fail fast)
# ------------------------------------------------------------------
echo "=========================================="
echo "Pre-flight: Running scorer & end-to-end unit tests"
echo "=========================================="
pytest tests/test_scoring.py tests/test_end_to_end.py -v --tb=short
echo "✅ Pre-flight tests passed"

# ------------------------------------------------------------------
# 3. Start vLLM server  (sections renumbered; pre-flight is 2)
# ------------------------------------------------------------------
echo "Starting vLLM server (TP=$TENSOR_PARALLEL_SIZE)..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs 512 \
    --enable-chunked-prefill \
    --enforce-eager \
    --max-num-batched-tokens 32768 \
    --disable-custom-all-reduce \
    &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

cleanup() {
    echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 4. Wait for readiness
# ------------------------------------------------------------------
echo "Waiting for vLLM server..."
MAX_WAIT=900
WAIT_INTERVAL=15
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ ERROR: vLLM server exited unexpectedly"
        exit 1
    fi
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM server ready after ${ELAPSED}s"
        break
    fi
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ ERROR: vLLM server timeout."
    exit 1
fi

# ------------------------------------------------------------------
# 5. Smoke Test Executions
# ------------------------------------------------------------------
echo "=========================================="
echo "Phase 1: Smoke Test Execution (--limit 5)"
echo "=========================================="
python -m experiments.run_phase1 --generator-model "$SERVED_MODEL_NAME" --limit 5 --max-workers 16
python -m analysis.plots --phase 1

echo "=========================================="
echo "Phase 2: Smoke Test Execution (--limit 5)"
echo "=========================================="
python -m experiments.run_phase2 --generator-model "$SERVED_MODEL_NAME" --limit 5 --max-workers 16
python -m analysis.plots --phase 2

echo "✅ Smoke test completed perfectly at $(date)"
