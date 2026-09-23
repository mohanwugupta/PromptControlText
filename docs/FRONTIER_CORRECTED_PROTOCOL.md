# Corrected frontier extension, version 1

One model per provider: OpenAI Astra first, then one Anthropic and one Google
model after their settings and costs are checked. Original judging is delegated
to the collaborator's cluster. The nano micro-pilot is not used for production
labels. No historical generation artifacts or judge implementations are changed.

## Frozen sample and interpretation

Seed 20260923; 100 sampled items, all 72 registry_v3 prompts plus a no-controller
baseline: 7,300 generation requests per model. Every provider and matched
historical-model rerun must use the same manifest, items, and condition order.

- HarmBench: 30 items (15 standard, 8 copyright, 7 contextual). ContextString is
  restored before Behavior, separated by two newlines. Context is not a system
  instruction. This is a versioned adapter convention.
- XSTest: 40 items (20 safe, 20 unsafe), balanced over prompt types within label.
  Existing prompt texts and labels are preserved.
- IHEval: 30 items (15 aligned, 15 conflict), sampled from the native
  system-prompt-extraction task, aligned/default and conflict/weak_defense.
  Each stratum samples 15 distinct parent defenses uniformly and then one probe
  per selected defense. Original system and user fields are preserved.

IHEval source: ytyz1307zzh/IHEval commit
`726a62924c3050045954df94347d53fe2bd1090d`; exact source paths and SHA256 checksums
are in the runner. These native pools contain 318 aligned and 438 conflict probes,
not the old 2,520 flattened raw probes. This is a narrowed, corrected extension,
not a claim to reproduce all native IHEval tasks. Native expected answers remain
available for later task-correctness scoring, separate from policy labels.

The model's system message consists of the native system instruction, followed
by two newlines and the selected registry controller. No-controller retains
IHEval's native system instruction. HarmBench/XSTest no-controller omits the
system role. Do not call the IHEval baseline "no system prompt". The intervention
composition and native subset must be reviewed when interpreting results.

The sample is disproportionate: report benchmark/stratum results, not an
unweighted original-dataset overall rate. Public metadata records population and
sample counts. For IHEval use parent-defense sampling and within-parent probe
probabilities appropriately; only one probe is selected per parent. Confidence
intervals must account for repeated conditions on the same item. Twenty safe
XSTest items give limited precision for false-refusal/SDT estimates.

## Astra settings and budget

Responses API with model `gpt-6-astra`, low reasoning, 2,048 combined reasoning/output
tokens, no temperature or seed, store=false, Batch processing. Returned model IDs,
usage, truncation status, policy blocks, and missing outputs are recorded.
This differs from the historical non-reasoning 512-token protocol. Do not claim
identical decoding across model families; match task inputs and report settings.

Astra cap: $75, including conservative reservations for pending and unresolved
requests. The full three-provider study cap remains $250. Reserve at most $75
for each later provider, leaving $25 for prior pilots and setup/contingency.
No paid OpenAI judge calls or Runpod provisioning are required. This local cap
is not a provider credit-balance check. The previously funded $10 account may
need a further top-up to complete Astra; never purchase credits automatically.

First submit the 100 no-controller baselines to check input handling and output
length before more conditions. Review the first batch for incomplete outputs;
if settings need changing, create a new run version and account for rerun costs.
Subsequent submissions contain at most 1,000 requests and must fit the remaining
worst-case cap. The runner allows only one outstanding batch. Completion windows
can be up to 24 hours. Failed/unknown POSTs are never auto-retried; `poll` attempts
read-only reconciliation by unique batch metadata. Unknown costs remain reserved.

API policy blocks are separate from generated refusals. Later unsubmitted
conditions for a blocked item are skipped; already submitted calls may finish.
Retain these rows and do not replace blocked items or rewrite prompts to evade
restrictions. Keep the frozen sample when comparing providers and report provider
missingness. There is no automatic retry for API errors or missing terminal rows.

## Commands

Python 3.12 plus PyYAML is sufficient for preparation and Astra generation.
Credentials stay in the existing `../.env.frontier` outside the repository.

```sh
python3 experiments/run_frontier_sample.py download-sources
python3 experiments/run_frontier_sample.py prepare
python3 experiments/run_frontier_sample.py submit --count 100
python3 experiments/run_frontier_sample.py poll
# Only after the current batch is terminal and reconciled:
python3 experiments/run_frontier_sample.py submit --count 1000
python3 experiments/run_frontier_sample.py poll
```

Only `submit` creates paid work. Source downloads and GET polling do not create
inference. `prepare`/`export` work offline after sources are downloaded.

`artifacts/frontier_sample_v1/summary.json` is the public progress record.
`judge_input.private.csv` includes only completed, nonempty responses, in stable
manifest order, and can be loaded by the original policy runner.
`all_statuses.private.csv` includes all 7,300 planned rows and their statuses.
`messages_json` records exact system/user messages; `input_text` retains the user
turn. The private manifest contains inputs, sampling, native answers, and tasks.
Original-model and later-provider generation must use messages_json or the
manifest, not input_text alone, or the IHEval defect would recur.

Raw outputs, inputs, native source files, state, and handoff CSVs are gitignored.
Do not publish native benchmark answers or response text as part of progress
tracking. Public checkouts recreate inputs from pinned sources only before a new
run; published results without private state cannot be resubmitted inadvertently.

## Original judge handoff

All eight checked-in judge manifests and `slurm/run_llm_judge.sh` identify
`meta-llama--Llama-3.1-8B-Instruct`, not a 70B checkpoint. Have the collaborator
confirm their actual original checkpoint before launching their cluster job.
The default wrapper reflects the recorded 8B model; JUDGE_MODEL must match the
served name of that same checkpoint. Do not change the judge based on memory.

Copy the finished `judge_input.private.csv` to the collaborator's cluster and
start their original vLLM judge service. Then:

```sh
bash scripts/judge_frontier_sample.sh /path/to/judge_input.private.csv
```

The wrapper preserves A/B/C prompts, original adjudicator, v1 schema,
temperature 0, max_tokens 250, and resume behavior. Existing code sends only
model_output to the judge; added benchmark metadata is not passed to it.
No judge is launched on this machine. Do not interpret unjudged generation as
safety or policy-performance results.

## Matched historical-model reruns

The collaborator should regenerate the frozen 100-item/73-condition manifest on
the original four checkpoints using the exact composed system/user messages.
Use distinct corrected-v1 job IDs and output directories; retain original artifacts.
Record checkpoint, chat template, output cap, and decoding settings. Score all
new outputs through the same confirmed original judge. This is a required plan,
not completed cluster work; cluster compute is outside the API spending estimate.

A cluster adapter is provided for these matched reruns. It uses the existing
vLLM client with caching/retries disabled, an explicit output cap, resumable
per-task records, and a separate output directory per checkpoint. For example:

```sh
python3 experiments/run_matched_frontier_sample.py \
  --manifest /path/to/manifest.private.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --max-tokens 512 \
  --output-dir /cluster/scratch/frontier_corrected_v1/llama31_8b
```

Repeat with each original generator checkpoint and a distinct directory. The
512-token example preserves its historical cap; it is not equivalent to Astra's
2,048 combined reasoning/output-token cap. Report the distinction. Set JOB_ID
to a distinct name when passing each resulting judge_input.csv to the judge wrapper.

## Reproduce the handoff inputs from a public checkout

```sh
python3 experiments/run_frontier_sample.py download-sources
python3 experiments/run_frontier_sample.py rebuild-inputs
```

This reconstructs all 100 corrected items and 7,300 task definitions, verifies
that the resulting public manifest is identical to the frozen archive, and
sends no paid requests. It creates manifest.private.json for matched cluster
reruns. It does not reconstruct model-generated responses, private batch IDs,
or the spending ledger. Resuming Astra requires its original private state;
the public results guard prevents paying for the archived requests again.
