#!/usr/bin/env bash
# Run on collaborator's cluster with the original judge's vLLM server already running.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_CSV="${1:-$PROJECT_DIR/artifacts/frontier_sample_v1/judge_input.private.csv}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama--Llama-3.1-8B-Instruct}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"
JOB_ID="${JOB_ID:-frontier_corrected_v1_astra}"
cd "$PROJECT_DIR"
python3 -m scoring.llm_policy_runner \
  --input "$INPUT_CSV" \
  --job-id "$JOB_ID" \
  --output-dir "artifacts/llm_policy_labels/$JOB_ID" \
  --model "$JUDGE_MODEL" \
  --base-url "$JUDGE_BASE_URL" \
  --prompt-a scoring/llm_policy_judge_prompt_A_v1.txt \
  --prompt-b scoring/llm_policy_judge_prompt_B_v1.txt \
  --prompt-c scoring/llm_policy_judge_prompt_C_v1.txt \
  --adjudicator-prompt scoring/llm_policy_adjudicator_prompt_v1.txt \
  --batch-size 512 --temperature 0.0 --max-tokens 250 --max-workers 256 --resume
