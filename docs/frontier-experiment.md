# Frontier extension: preparation and execution record

Preparation status (October 7, 2026, Los Angeles): **offline preparation validated**.
No inference calls or paid Runpod resources were created during preparation.
The subsequent [live Runpod pilot](../artifacts/frontier/runs/2026-10-08-pilot/README.md)
completed 50 OpenAI cases and stopped the other providers on billing errors.
The worker was deleted after verified result retrieval; 100 cases remain unfinished.
Three subsequent Anthropic retries also returned credit-balance errors; see the
[latest pilot record](../artifacts/frontier/runs/2026-10-08-anthropic-billing-check/README.md)
and the [overall revision status](revision-status-2026-10-07.md).
API model-list checks confirmed access to the three candidate IDs below; they do
not establish inference permissions, funded billing, or usable rate limits.

## Frozen design

The extension uses one model per provider, the same 50 items on all three models,
and all 73 existing conditions: 72 registry-v3 interventions plus the control.
This gives **10,950 main responses**. The frozen manifest contains full inputs,
references, sample IDs, prompt text, source hashes, and a content hash. No model
outcomes were used to choose items.

| Stratum | Main | Disjoint pilot |
|---|---:|---:|
| XSTest safe | 20 | 2 |
| XSTest unsafe | 10 | 2 |
| HarmBench | 10 | 2 |
| IHEval aligned | 5 | 2 |
| IHEval conflict | 5 | 2 |
| Total | 50 | 10 |

Sampling uses SHA256 ranks with seed `frontier-2026-10-07-v1`, fixed stratum
order, and exclusions for repeated IHEval parents and exact duplicate user
texts. The pilot excludes both main-sample parents and user texts. This is an
item sample with those exclusions, not a claim of uniform parent sampling.
The resulting main HarmBench sample has 6 standard, 3 contextual, and 1 copyright
item. Copyright behavior must be reported separately from physical/other harmful
compliance. The pilot has one standard and one contextual HarmBench item.

The pilot crosses its 10 items with the control and explicit/v1 Answer-first,
Refuse-first, Source-isolation-first, and Hierarchy-first conditions on all three
models: **150 pilot requests**, separate from the 10,950 main responses. It tests
API compatibility, cost, truncation and logging; it does not establish policy
effects or validate the judge. Freeze any revised generation settings before
main generation. Do not select new sample items after seeing comparative results.

## Benchmark context repairs

The original loaders omit required context. The new `frontier` package preserves
the legacy runner and historical outputs, and prepares the corrected extension:

- HarmBench includes `ContextString` before `Behavior`, separated by
  `\n\n---\n\n`, matching the [official DirectRequest implementation](https://github.com/centerforaisafety/HarmBench/blob/8e1604d1171fe8a48d8febecd22f600e462bdcdd/baselines/direct_request/direct_request.py).
- IHEval restores task system instructions and reference answers from the
  [official prepared safety inputs](https://github.com/ytyz1307zzh/IHEval/tree/726a62924c3050045954df94347d53fe2bd1090d/benchmark/safety).
  The union of both safety subtasks, default aligned inputs and weak-defense
  conflict inputs recovers all 2,520 historical IDs and user texts, allowing
  only CRLF-to-LF normalization. Per-item metadata records that normalization.
  This extends the repository's safety subset, not the entire IHEval benchmark.
- For IHEval, the unchanged task system comes first, then two newlines and the
  experimental intervention in the same system field. The control retains the
  task system but adds no intervention. On the other benchmarks, the control
  omits the system field entirely. This distinction must appear in the paper.

Do not pool the corrected contextual tasks with old context-incomplete results
as though only model identity changed. Paired within-item comparisons within
this extension are the primary analysis.

## Candidate models and fixed pilot settings

| Provider | Requested ID | Input / output USD per million tokens |
|---|---|---:|
| OpenAI | `gpt-6-astra` | 10 / 50 |
| Anthropic | `claude-opus-5-5` | 4 / 20 |
| Google | `gemini-3.1-pro-preview` | 2 / 12 |

Sources checked October 8 UTC / October 7 Pacific:
[OpenAI model/pricing](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Anthropic models/pricing](https://platform.claude.com/docs/en/models/overview),
[Google pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-pro-preview).
These are standard short-context prices; the runner uses no paid tools or
explicit cache creation. Conservative accounting reserves 1.25x standard input
pricing for OpenAI and Anthropic to cover possible cache writes. This can exceed
the actual invoice, particularly for cached inputs. No Batch discount is assumed.

All three use low reasoning effort and a 4,096 output-token limit. The label
"low" does not represent matched compute across providers. Sampling parameters
are omitted for OpenAI and Anthropic; Gemini uses temperature 1.0. Claude Opus
5.5 no longer accepts non-default sampling parameters, and Gemini recommends
its default temperature: see the [Claude migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)
and [Gemini 3 guide](https://ai.google.dev/gemini-api/docs/gemini-3).
This is not a temperature-zero replication of the old models.

OpenAI uses Responses, Anthropic uses Messages, and Google uses generateContent.
Only the canonical system/user text is sent; no output-format instructions,
tools, safety-setting overrides, or conversation history are added. Provider
defaults remain part of the evaluated API configuration. Returned model/version,
request ID, token usage, finish reason, and safety blocks are recorded. Model
aliases and preview versions can change: compare returned versions across the
run and report any drift. Anthropic documents its selected ID as fixed.

## Run locally or on a Runpod CPU worker

From the repository root, use a Python 3.9+ environment:

```bash
python -m pip install -r requirements-frontier.txt
python -m frontier.prepare --download
python -m frontier.run
python -m frontier.run --split main
```

These are preparation/dry-run commands and make no inference calls. `--download`
fetches only the four pinned public IHEval source files into `.local/iheval`.
The committed manifest is sufficient for running: it does not require that cache.
Rebuilding the manifest refuses to overwrite different bytes.

To execute the **paid pilot**, with the three provider keys in environment
variables or the ignored repository `.env`:

```bash
python -m frontier.run --execute
python -m frontier.report
```

The default pilot ledger is `.local/frontier/pilot.sqlite`; its JSONL export is
`.local/frontier/pilot.jsonl`. Preserve both. A completed request is not reissued.
SQLite persists a cost reservation before each network call. Missing usage,
unknown outcomes, errors, and an interrupted attempt retain their reservation
and stop execution for reconciliation. There are no automatic retries. Do not
delete the ledger, start a second ledger to evade its cap, or blindly resend a
timed-out request. If an interruption leaves a reservation, first recover the
provider's response/usage or record the possible charge before a separate,
explicitly documented retry implementation.

An observed OpenAI HTTP `cyber_policy` block is recorded as `blocked`, with an
empty assistant answer and its full cost reservation retained. It is not retried
or labeled as a textual refusal. `--providers openai` (or another explicit subset)
can continue unaffected providers in the same ledger; unresolved costs from all
providers still count toward the shared cap. This does not make a partial pilot
complete. Billing failures remain blocked until the account state changes.

After the user confirms Anthropic credits have been added, a known HTTP 400
credit-balance rejection can be reopened explicitly:

```bash
python -m frontier.reconcile --request-id REQUEST_ID --funding-update "User confirmed credits added"
python -m frontier.run --execute --providers anthropic
```

Back up the SQLite ledger first. Reconciliation atomically moves that failed
attempt into `attempt_history`, retaining its complete record and cost, and
makes only the unchanged request eligible for a new attempt. Other failures,
completed outputs, and safety blocks are ineligible. There are no automatic
retries. The budget includes both current and archived attempts, including if
execution stops between reconciliation and the next request. JSONL exports
embed `prior_attempts` on each case; reports include their costs separately from
the current-response cost projection. Preserve the same ledger on the next pod.

The live pilot has a **$10 total cap**: $9 for API accounting and $1 reserved for
Runpod infrastructure, part of the $60 setup/contingency allocation.
Each call reserves 32,768 input tokens plus its maximum output at conservative
rates. Oversized request bodies are rejected before calls. Actual returned usage
releases unused reservation. This is conservative software accounting, not a
provider billing guarantee; verify invoices and current prices. Runpod rental,
other account activity, prior spending, and judging are outside this ledger.
Record those separately against the overall $250 ceiling. The remaining planned
allocations are $160 generation and $30 judging; these are allocations, not
measured estimates. A cap stop may leave the pilot incomplete.

Runpod deployment uses a small CPU pod to call the
three hosted APIs; a GPU does not accelerate those calls. Before launch:

1. Inspect current CPU pricing/capacity and the installed CLI's live help.
2. Register a project SSH public key before pod creation. Keep its private key
   local and ignored. Provision a current official Python-capable image with
   checkpoint storage and a cleanup deadline.
3. Check out the exact preparation commit, install the lightweight requirements,
   and transfer only the three provider keys over SSH into a mode-600 `.env`.
   The worker does not need the Runpod control key or HF token.
4. Run both dry runs on the pod, then the capped pilot. Retrieve the ledger,
   export, and summary, and verify them locally before terminating resources.
5. Record the pod ID, image, commit, hourly price, timestamps and rental spend.
   Publish reviewed experiment artifacts and run summaries to GitHub; keep all
   keys, private SSH files, and raw environment dumps out of Git.

Runpod CLI 2.14.0's observed `pod create --help` does **not** expose the
`--terminate-after` option mentioned in the plugin's generic guide. The legacy
GraphQL creation method rejected the CPU request with `gpuTypeId is required`.
The live pilot therefore uses the supported REST CPU creation fields and a local
three-hour API deletion watchdog (`frontier/runpod_watchdog.py`), with host idle
sleep inhibited. This is a local safeguard, not a claimed server-side schedule.
The selected worker has 2 vCPUs, 4 GB RAM, a 10 GB container disk, no network
volume, and a $0.06/hour compute rate. SQLite-consistent checkpoints are copied
to the local host during execution and all results are retrieved before deletion.
Only SSH is exposed. No web inference service is deployed.

## Readiness and remaining scientific work

Offline checks pass: 50/10 disjoint items, complete context, 73 conditions,
10,950/150 unique requests, consistent provider payload text, token accounting,
budget stops, provider-block handling, crash reservations, and duplicate-free
resume in mocked calls. See `artifacts/frontier/preparation-status.json`.

The runner intentionally refuses paid `--split main`. After the pilot, inspect
errors and truncations, estimate cost by provider and stratum, and reweight to
the main quotas. The report supplies 50% planning headroom; it is not a statistical
upper bound and five pilot conditions cannot perfectly predict 73. Only then
create/freeze the main execution configuration and transport (Batch can be
considered if required by measured costs). Main readiness also requires a
working Runpod deployment and a reconciled study budget.

Keep **`meta-llama/Llama-3.1-8B-Instruct`** as the judge with the existing A/B/C
rubrics and adjudication. These are output-only judgments by design; do not add
task context or substitute a different judge while claiming the same rater.
Snapshot the judge prompt files/settings and verify row-level coverage before
judging: the legacy resume code keys on response-text hashes, so identical
responses need an explicit mapping back to every generation request.
Human-validate labels on current-model outputs. Native IHEval correctness and
harmful-compliance correctness are separate from the output-only six-policy label.
Provider blocks, transport failures and truncated outputs need separate reporting,
not automatic refusal labels. Use a GPU pod or Mohan's cluster only for judging.

Analyze policy transitions with the same item across conditions. Cluster
uncertainty by item/parent, report exploratory scope, and avoid precise subgroup
rankings from five IHEval items per setting. Define the positive class and endpoint
labels before computing SDT d-prime and criterion; non-refusal is not automatically
harmful compliance. Keep the supplied related-work rewrite focused on this combined
within-item intervention, six-policy taxonomy, and safety-tradeoff analysis.
