#!/bin/bash
# =============================================================================
# Submit smoke-test + full jobs for all models in the registry (or a subset).
#
# Usage:
#   slurm/submit_all_models.sh                # all 7 models
#   slurm/submit_all_models.sh gemma4_12b qwen3_6_35b_a3b   # subset
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_SLUGS=(
    gemma4_12b
    gemma4_31b
    qwen3_6_35b_a3b
    deepseek_r1_distill_qwen_32b
    deepseek_r1_distill_llama_70b
    nemotron_3_5_lightning_30b_a3b
    nemotron_3_nano_4b_gguf
)

if [ "$#" -gt 0 ]; then
    SLUGS=("$@")
else
    SLUGS=("${ALL_SLUGS[@]}")
fi

for slug in "${SLUGS[@]}"; do
    echo "=========================================="
    echo "$slug"
    echo "=========================================="
    "$SCRIPT_DIR/submit_model.sh" "$slug"
    echo ""
done
