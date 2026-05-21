#!/bin/bash
#SBATCH --job-name=llm_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --array=0-5
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/llm_judge_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/llm_judge_%A_%a.err

# =============================================================================
# LLM Policy Judge — output-only behavioral classifier (PRD §19 / §20)
#
# Runs as a 6-task SLURM array (--array=0-5).  Each task starts its own
# vLLM server and processes one phase1 CSV through the 3-pass judge pipeline.
#
# Judge model: Llama-3.1-8B-Instruct
#   - already on cluster, no startup issues, strong JSON instruction following
#   - NOT gpt-oss-20b (openai_harmony startup failures + circularity risk)
#
# Approximate throughput: ~150 req/s → 7.8M calls (largest CSV) ≈ 14–15 h
# All tasks use --resume so they can be safely resubmitted if they time out.
#
# Submit: sbatch slurm/run_llm_judge.sh
# Resubmit (resume): sbatch slurm/run_llm_judge.sh   (--resume skips done rows)
# Single task only:  sbatch --array=4 slurm/run_llm_judge.sh   (0=qwen25_72b … 5=control_qwen2_7b)
# =============================================================================

set -eo pipefail

# ------------------------------------------------------------------
# 0. Per-task job registry  (index → job_id | input CSV | output dir)
# ------------------------------------------------------------------
JOB_IDS=(
    "phase1_qwen25_72b"
    "phase1_llama31_8b"
    "phase1_llama33_70b"
    "phase1_qwen2_7b"
    "control_llama31_8b"
    "control_qwen2_7b"
)
INPUT_CSVS=(
    "artifacts/phase1_results.csv"
    "artifacts/phase1_results_llama31_8b.csv"
    "artifacts/phase1_results_llama33_70b.csv"
    "artifacts/phase1_results_qwen2_7b.csv"
    "artifacts/phase1_results_control_llama31_8b.csv"
    "artifacts/phase1_results_control_qwen2_7b.csv"
)

TASK=${SLURM_ARRAY_TASK_ID}
JOB_ID="${JOB_IDS[$TASK]}"
INPUT_CSV="${INPUT_CSVS[$TASK]}"
OUTPUT_DIR="artifacts/llm_policy_labels/${JOB_ID}"

# Each array task gets its own port so tasks can run on the same node if needed
VLLM_PORT=$((8000 + TASK))

echo "=========================================="
echo " LLM Policy Judge — task ${TASK} / ${JOB_ID}"
echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $TASK"
echo "Node:        $SLURMD_NODENAME"
echo "Time:        $(date)"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Input:       $INPUT_CSV"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/PromptControlText
MODEL_DIR_NAME=meta-llama--Llama-3.1-8B-Instruct
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/$MODEL_DIR_NAME
SERVED_MODEL_NAME=$(basename "$MODEL_PATH")
CONDA_ENV=PromptControlText
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

# Judge call settings
BATCH_SIZE=512
MAX_WORKERS=256
TEMPERATURE=0.0
MAX_TOKENS=250

# ------------------------------------------------------------------
# 2. Environment setup
# ------------------------------------------------------------------
cd "$PROJECT_DIR"

module load anaconda3/2025.6

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f "$HOME/.conda/envs/$CONDA_ENV/bin/activate" ]; then
    source "$HOME/.conda/envs/$CONDA_ENV/bin/activate"
else
    source activate "$CONDA_ENV"
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------
# 3. Cache & offline settings
# ------------------------------------------------------------------
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_DISK_DIR=$HF_DATASETS_CACHE
export VLLM_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache
export VLLM_USAGE_STATS_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/usage_stats
export TRITON_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/triton
export XDG_CACHE_HOME=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/xdg
export TIKTOKEN_CACHE_DIR=$HOME/.tiktoken_cache

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_DIR" "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------------
# 4. GPU / Memory optimizations
# ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ------------------------------------------------------------------
# 5. Validate prerequisites
# ------------------------------------------------------------------
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Judge model found: $MODEL_PATH"
else
    echo "❌ ERROR: Judge model not found at: $MODEL_PATH"
    exit 1
fi

if [ -f "$PROJECT_DIR/$INPUT_CSV" ]; then
    ROW_COUNT=$(wc -l < "$PROJECT_DIR/$INPUT_CSV")
    echo "✅ Input CSV found: $INPUT_CSV ($ROW_COUNT lines including header)"
else
    echo "❌ ERROR: Input CSV not found: $PROJECT_DIR/$INPUT_CSV"
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/$OUTPUT_DIR"

# ------------------------------------------------------------------
# 6. Start vLLM server
# ------------------------------------------------------------------
echo ""
echo "Starting vLLM server ($SERVED_MODEL_NAME, TP=$TENSOR_PARALLEL_SIZE, port=$VLLM_PORT)..."

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
# 7. Wait for server readiness
# ------------------------------------------------------------------
echo "Waiting for vLLM server on port $VLLM_PORT..."
MAX_WAIT=600
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
    echo "❌ ERROR: vLLM server failed to start within ${MAX_WAIT}s"
    exit 1
fi

# ------------------------------------------------------------------
# 8. Run LLM Policy Judge
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo " LLM Judge: ${JOB_ID}"
echo " Input:     ${INPUT_CSV}"
echo " Output:    ${OUTPUT_DIR}"
echo " Model:     ${SERVED_MODEL_NAME}"
echo "=========================================="

python -m scoring.llm_policy_runner \
    --input          "$PROJECT_DIR/$INPUT_CSV" \
    --job-id         "$JOB_ID" \
    --output-dir     "$PROJECT_DIR/$OUTPUT_DIR" \
    --model          "$SERVED_MODEL_NAME" \
    --base-url       "http://localhost:${VLLM_PORT}/v1" \
    --prompt-a       "$PROJECT_DIR/scoring/llm_policy_judge_prompt_A_v1.txt" \
    --prompt-b       "$PROJECT_DIR/scoring/llm_policy_judge_prompt_B_v1.txt" \
    --prompt-c       "$PROJECT_DIR/scoring/llm_policy_judge_prompt_C_v1.txt" \
    --adjudicator-prompt "$PROJECT_DIR/scoring/llm_policy_adjudicator_prompt_v1.txt" \
    --batch-size     "$BATCH_SIZE" \
    --temperature    "$TEMPERATURE" \
    --max-tokens     "$MAX_TOKENS" \
    --max-workers    "$MAX_WORKERS" \
    --resume

echo ""
echo "✅ Task ${TASK} (${JOB_ID}) completed at $(date)"
