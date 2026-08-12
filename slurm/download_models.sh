#!/bin/bash
# =============================================================================
# Download HuggingFace models onto the shared cluster model directory.
#
# Run this on the LOGIN NODE (or any node with internet access) — NOT via
# sbatch on a compute node, since compute nodes are typically firewalled off
# from the public internet on most HPC clusters (e.g. Princeton della/adroit).
#
# Usage:
#   ./slurm/download_models.sh                      # download everything
#   ./slurm/download_models.sh gemma4_12b qwen3_6_35b_a3b   # only these slugs
#
# Prereqs:
#   - conda env "PromptControlText" has huggingface_hub installed
#   - export HF_TOKEN=<your token> (needed for gated models: Gemma, Nemotron)
#   - You've clicked "Agree" on the license page for any gated repo on
#     huggingface.co before downloading.
# =============================================================================

set -eo pipefail

PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/PromptControlText
MODELS_ROOT=/scratch/gpfs/JORDANAT/mg9965/models
CONDA_ENV=PromptControlText

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
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
mkdir -p "$HF_HOME" "$MODELS_ROOT"

if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN is not set. Gated models (Gemma, Nemotron) will fail to download."
    echo "    export HF_TOKEN=hf_xxx   # or run: huggingface-cli login"
fi

if [ "$#" -gt 0 ]; then
    python -m models.download_models --models-root "$MODELS_ROOT" --only "$@"
else
    python -m models.download_models --models-root "$MODELS_ROOT"
fi

echo "✅ Done. Models are in $MODELS_ROOT"
