# Judge the completed frontier main experiment on the cluster

The completed main upload is in
`artifacts/frontier/runs/2026-10-08-main/responses/*.jsonl`. It covers all
10,950 frozen requests: 50 items × 73 conditions × three providers. The pilot
is a separate experiment and is excluded from this workflow.

The adapter uses the existing output-only Llama-3.1-8B-Instruct judge, A/B/C
rubrics, adjudication rules, and six policy labels. Only `model_output` enters
the judge. Item text, gold labels, references, prompt conditions, model names,
and provider flags remain analysis metadata. Native task correctness must be
scored separately with the corrected benchmark context.

## Prepare and submit

From the cluster repository, using the existing `PromptControlText` environment:

```bash
cd /scratch/gpfs/JORDANAT/mg9965/PromptControlText
git pull --ff-only
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate PromptControlText
python -m frontier.prepare_judge
mkdir -p logs
sbatch slurm/run_frontier_judge.sh
```

Preparation is offline and requires no API credentials or GPU. It checks the
frozen design, complete request coverage, request and response hashes, and
resolved generation states before replacing prepared inputs. Four historical
Gemini `needs_review` records are accepted only through their explicit terminal
block resolution in the published progress file.

The generated inputs and job registry live in `.local/frontier/judge-main/`,
which is Git-ignored. Prepare these files on the cluster after pulling; the
launcher refuses changed hashes, incomplete inputs, or duplicate requests
before starting vLLM.

| Array task | Generator | All cases | Judgeable answers | Empty provider blocks |
|---|---|---:|---:|---:|
| 0 | OpenAI `gpt-6-astra` | 3,650 | 3,650 | 0 |
| 1 | Anthropic `claude-opus-5-5` | 3,650 | 3,181 | 469 |
| 2 | Google `gemini-3.1-pro-preview` | 3,650 | 3,609 | 41 |
| Total | | 10,950 | 10,440 | 510 |

Empty provider blocks retain their transport outcome and have no policy label.
They are not automatically labeled refusal. A blocked result with nonempty
text still goes through the judge. No-system controls and prompted conditions
are both included, with `experiment_group` preserved.

Each array task requests one 80 GB GPU, serves the existing local Llama model
from `/scratch/gpfs/JORDANAT/mg9965/models/meta-llama--Llama-3.1-8B-Instruct`,
and uses the `PromptControlText` environment. It defaults to 32 concurrent
requests and ports 8100–8102. Each answer receives three first-pass votes,
plus adjudication when required. Allow up to the 24-hour job limit.

## Resume and collect

Resubmit the same script to resume; or retry a particular provider:

```bash
sbatch slurm/run_frontier_judge.sh
sbatch --array=1 slurm/run_frontier_judge.sh
```

Completed resolutions are checkpointed by request ID and answer hash. Identical
answers from different requests keep separate labels; saved adjudication is
replayed without model calls. Changing judge settings or rubric text requires
a separate output directory. Concurrent writers to one job directory are
rejected. If the server becomes unreachable, the runner stops and preserves
completed checkpoints for the next submission.

Per-provider outputs are written to:

```text
artifacts/llm_policy_labels/frontier_main_openai/
artifacts/llm_policy_labels/frontier_main_anthropic/
artifacts/llm_policy_labels/frontier_main_google/
```

Each includes `labeled.csv`, `labels_only.csv`, `judge_votes.csv`, any
`adjudication_votes.csv`, `resolutions.jsonl`, prompt copies, settings, summaries,
an audit sample, and `manifest.json`. A job succeeds only when every judgeable
answer has a valid policy label; parse failures must be retried.

If every row was processed but `manifest.json` has `complete=false`, the job
still contains invalid judgments. The merge reports the valid-label count
separately from model, job-ID, or input-hash mismatches. Resubmission retries
only requests without valid saved labels. An early validation bug rejected
adjudications whose explanations exceeded 280 characters; explanatory text is
now bounded before schema validation, while labels and other fields remain
strictly validated. Pull the fix before retrying those cases. The original
generation does not need to be repeated.

After all three tasks succeed:

```bash
python -m frontier.finish_judge
```

This verifies each output's request coverage, unchanged input metadata, canonical
judge model, schema version, and input hash, then joins labels by request ID.
The resulting `.local/frontier/judge-main/labeled_all_cases.csv` contains all
10,950 cases: 10,440 judged answers and 510 empty blocks with blank labels and
`judge_status=not_applicable_empty_provider_block`. The original normalized
responses remain the generation record. Keep this context-repaired frontier
experiment separate from the older context-incomplete IHEval runs in analysis.

## Local validation

Offline tests cover the actual full upload, conversion/hash failures,
identical-answer and reordered-input resumes, saved adjudication, interrupted
checkpoint appends, changed settings, incomplete labels, disconnected servers,
and joining judged answers with empty blocks. Cluster GPU execution must be
verified after submission; these tests do not load the Llama model.

```bash
python -m pytest -q tests/test_frontier_judge.py tests/test_llm_policy_*.py
bash -n slurm/run_frontier_judge.sh slurm/run_llm_judge.sh
```
