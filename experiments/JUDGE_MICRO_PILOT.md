# Cheapest-judge micro-pilot

A deliberately small feasibility test: GPT-5.4 nano labels 12 existing frontier
responses using each of the original A/B/C judge prompts (36 requests total).
Selection uses two distinct items per benchmark/source-model stratum and seed
20260922, then selects one available condition per item. The sample is not
representative and does not guarantee coverage of all six behavioral labels.

The judge receives only the assistant response, never source-model identity,
benchmark, original prompt, or gold labels. It uses strict structured output
with the existing schema, no reasoning, a 512-token cap, and standard pricing
($0.20 input / $1.25 output per million tokens). No adjudicator is run.

```sh
python3 experiments/run_judge_micro_pilot.py
python3 experiments/run_judge_micro_pilot.py --execute --limit 1
python3 experiments/run_judge_micro_pilot.py --execute
python3 -m unittest discover -s tests -p 'test_judge_micro_pilot.py'
```

Only `--execute` sends paid requests. Credentials come from `../.env.frontier`.
The adapter reuses `run_api_pilot.py` transport, durable ledger, locking, and
conservative retry controls, with its own directory and a $1 cap. There are no
automatic retries of uncertain calls. The first execution command can check
schema compatibility before running the remainder. Completed tasks are skipped.

Inputs, raw judge evidence, and the ledger remain private and gitignored.
Public `artifacts/judge_micro_pilot/manifest.json` records sample IDs and source
hashes; `usage.jsonl` contains labels and costs without quotations. `summary.json`
and `REPORT.md` summarize results. Public checkouts lack private inputs and
cannot repeat the paid micro-pilot inadvertently. The source response archive
must remain unchanged to resume this run.

This is a schema/cost/agreement check, not an accuracy validation. Three prompts
on one model produce correlated votes. A/B/C also differ in their rules for
mixed refusal-and-answer responses. Human review against a reconciled rubric is
needed before adopting nano as a production judge. Model confidence is not a
calibrated measure of correctness. Original benchmark labels are not ground
truth for the six output-only behavioral labels.

Sources: [nano model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-nano)
and [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
