#!/bin/bash
#SBATCH --job-name=frontier_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/frontier_judge_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/PromptControlText/logs/frontier_judge_%A_%a.err

# Prepare first: python -m frontier.prepare_judge
# Submit/resume: sbatch slurm/run_frontier_judge.sh
# Indices: 0=OpenAI, 1=Anthropic, 2=Google. Same Llama judge and A/B/C rubrics.
set -eo pipefail
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/PromptControlText
cd "$PROJECT_DIR"
export JUDGE_JOBS_CONFIG=.local/frontier/judge-main/jobs.yaml
export JUDGE_REQUIRE_COMPLETE=1
export JUDGE_MAX_WORKERS=${JUDGE_MAX_WORKERS:-32}
export JUDGE_PORT_BASE=8100
exec bash "$PROJECT_DIR/slurm/run_llm_judge.sh"
