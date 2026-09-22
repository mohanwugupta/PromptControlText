# Frontier API feasibility pilot

Run from the repository root with Python 3.12 and PyYAML installed:

```sh
python3 experiments/run_api_pilot.py
python3 experiments/run_api_pilot.py --execute
```

The first command only prepares/summarizes local files. The second sends paid requests.
Credentials are loaded from `../.env.frontier`; do not copy that file into this repository.
After fixing API billing or waiting for a rate limit to reset, `--retry-rejected`
permits retrying confirmed credit/rate-limit rejections.
Other failed or uncertain requests are not automatically retried.

The frozen manifest records repository commit, source hashes, sampling seed, exact
inputs and prompts, model IDs, and generation settings. It includes eight items
per benchmark, with XSTest split equally across safe/unsafe and IHEval split equally
across aligned/conflict. Each item has a no-system-message control and three
assigned registry prompts. All 72 registry prompts appear once per model, making
192 planned requests across two models. This is a sparse technical pilot, not the
fully crossed paper experiment and not sufficient for inferential safety claims.

Both models use low reasoning effort and a 2,048-token output budget including
reasoning. Temperature and seed are not supplied. These settings are not identical
to the historical 512-token non-reasoning configuration. Returned model IDs,
service tier, usage (including reasoning), response status, final text, and request
IDs are saved. Hidden reasoning text is not requested.

The append-only ledger reserves a conservative maximum cost before each request.
Confirmed usage settles the reservation; uncertain failures retain it. Three
requests may run concurrently, all sharing a $10 cap. An exclusive file lock
prevents simultaneous runner processes. Credentials and response text are never
printed to the terminal. Local response artifacts may contain harmful benchmark
material and are intended for research review.

Provider `cyber_policy` errors are recorded as API-level blocks, not generated
refusals. Remaining variants of the same benchmark item are skipped across both
models, without rewriting the item or retrying it under another prompt. Other
HTTP errors stop further dispatch. Connection interruptions pause dispatch for
20 seconds before moving to different tasks; three consecutive connection
failures stop the run. Unknown requests are never automatically retried.
Requests already in flight may finish. Requests are paced at least 3.1 seconds
apart. Skips mean planned prompt coverage can differ from realized coverage.

Artifacts are saved under `artifacts/frontier_pilot/`:

- `manifest.json`: frozen requests and provenance.
- `responses.jsonl`: append-only attempt history, including prior billing errors.
- `ledger.jsonl`: reservations and settlements.
- `summary.json`: cost and status summary.

Cost extrapolations cover generation only; judging, auditing, retries, and different
item/condition mixtures require additional allowances. Raw outputs remain unjudged
until an explicit validated scoring stage. In particular, the repository's IHEval
loader flattens source inputs and its HarmBench loader omits ContextString; this
pilot preserves that behavior for comparison but cannot validate the underlying
benchmark construction.

## Reproducing the published analysis without paid calls

```sh
python3 analysis/analyze_api_pilot.py
python3 analysis/forecast_api_cost.py
python3 -m unittest discover -s tests -p 'test_api_pilot.py'
```

`usage.jsonl` publishes only attempt metadata, usage, and output character counts.
`budget_accounting.json` publishes the aggregate conservative reservation total.
The report can be regenerated from these files without private raw completions.
A public checkout cannot resume paid requests without the original private ledger;
the runner refuses execution to prevent duplicating the archived pilot. A new
study needs a separate versioned output directory and a reviewed manifest.
The runner is a pilot tool, not yet a production or Batch API runner.
