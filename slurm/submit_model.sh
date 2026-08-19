#!/bin/bash
# =============================================================================
# Submit a smoke test + full vLLM serving/eval job for one of the new models.
#
# The smoke test job boots vLLM and sends a single chat completion request.
# The full job (run_model_generic.sh) is submitted immediately afterwards
# with --dependency=afterok:<smoke_job_id>, so Slurm will only start it once
# the smoke test has exited successfully (and will never start it — Slurm
# auto-cancels dependents — if the smoke test fails).
#
# Usage:
#   slurm/submit_model.sh <slug>
#
# Slugs (must match configs/model_registry.yaml):
#   gemma4_12b
#   gemma4_31b
#   qwen3_6_35b_a3b
#   deepseek_r1_distill_qwen_32b
#   deepseek_r1_distill_llama_70b
#   nemotron_3_5_lightning_30b_a3b
#   nemotron_3_nano_4b_gguf
# =============================================================================

set -eo pipefail

SLUG="$1"
if [ -z "$SLUG" ]; then
    echo "Usage: $0 <slug>"
    exit 1
fi

PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/PromptControlText
SMOKE_SCRIPT="$PROJECT_DIR/slurm/run_smoke_test.sh"
GENERIC_SCRIPT="$PROJECT_DIR/slurm/run_model_generic.sh"

# slug -> "MODEL_DIR_NAME|GPUS|TP|MAX_LEN|GPU_MEM_UTIL|IS_MOE|IS_GGUF|GGUF_FILE"
case "$SLUG" in
  gemma4_12b)
    MODEL_DIR_NAME="google--gemma-4-12b-it"; GPUS=1; TP=1; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE="" ;;
  gemma4_31b)
    MODEL_DIR_NAME="google--gemma-4-31b-it"; GPUS=2; TP=2; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE="" ;;
  qwen3_6_35b_a3b)
    MODEL_DIR_NAME="Qwen--Qwen3.6-35B-A3B-FP8"; GPUS=2; TP=2; MAX_LEN=8192; MEM=0.92; MOE=1; GGUF=0; GGUF_FILE="" ;;
  deepseek_r1_distill_qwen_32b)
    MODEL_DIR_NAME="deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"; GPUS=2; TP=2; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE="" ;;
  deepseek_r1_distill_llama_70b)
    MODEL_DIR_NAME="deepseek-ai--DeepSeek-R1-Distill-Llama-70B"; GPUS=4; TP=4; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=0; GGUF_FILE="" ;;
  nemotron_3_5_lightning_30b_a3b)
    MODEL_DIR_NAME="nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"; GPUS=2; TP=2; MAX_LEN=8192; MEM=0.92; MOE=1; GGUF=0; GGUF_FILE="" ;;
  nemotron_3_nano_4b_gguf)
    MODEL_DIR_NAME="nvidia--NVIDIA-Nemotron-3-Nano-4B-GGUF"; GPUS=1; TP=1; MAX_LEN=8192; MEM=0.92; MOE=0; GGUF=1
    # Adjust to the exact quant file you downloaded, e.g. *-Q4_K_M.gguf, *-Q8_0.gguf
    GGUF_FILE="nemotron-3-nano-4b-Q4_K_M.gguf" ;;
  *)
    echo "❌ Unknown slug: $SLUG"
    echo "   See configs/model_registry.yaml for valid slugs."
    exit 1 ;;
esac

mkdir -p "$PROJECT_DIR/logs"

EXPORT_VARS="ALL,MODEL_DIR_NAME=$MODEL_DIR_NAME,MODEL_SLUG=$SLUG,TENSOR_PARALLEL_SIZE=$TP,MAX_MODEL_LEN=$MAX_LEN,GPU_MEMORY_UTILIZATION=$MEM,IS_MOE=$MOE,IS_GGUF=$GGUF,GGUF_FILE=$GGUF_FILE"

echo "Submitting smoke test for $SLUG (GPUs=$GPUS, TP=$TP)..."
SMOKE_JOB_ID=$(sbatch --parsable \
    --job-name="pct_smoke_${SLUG}" \
    --gres="gpu:${GPUS}" \
    --export="$EXPORT_VARS" \
    "$SMOKE_SCRIPT")
echo "  -> smoke test job id: $SMOKE_JOB_ID"

echo "Submitting full job for $SLUG (dependent on smoke test succeeding)..."
FULL_JOB_ID=$(sbatch --parsable \
    --job-name="pct_${SLUG}" \
    --gres="gpu:${GPUS}" \
    --dependency="afterok:${SMOKE_JOB_ID}" \
    --export="$EXPORT_VARS" \
    "$GENERIC_SCRIPT")
echo "  -> full job id: $FULL_JOB_ID (will run only if job $SMOKE_JOB_ID succeeds)"
