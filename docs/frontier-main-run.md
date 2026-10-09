# Main frontier extension — frozen execution plan

Prepared October 8, 2026 UTC (October 7 Pacific). **Main generation has resumed
from the existing ledger.** All 1,615 earlier cases were recovered, the four
Gemini blocks retain their full cost reservations, and the next 600 cases were
accepted. See the [resume launch](../artifacts/frontier/runs/2026-10-08-main/resume-launch.json)
and [timestamped progress](../artifacts/frontier/runs/2026-10-08-main/progress.json).
The original sample, model settings, budgets and cleanup deadline are unchanged.

## Design and settings

The unchanged manifest crosses 50 items with 73 conditions and one generator
per provider: **10,950 cases**, 3,650 per model. Main quotas are XSTest safe 20,
XSTest unsafe 10, HarmBench 10, IHEval aligned 5 and conflict 5. The pilot uses
10 different items and is excluded from main analysis. Context repairs,
sampling, caveats and the 73 interventions are described in
[the experiment record](frontier-experiment.md).

Models remain `gpt-6-astra`, `claude-opus-5-5`, and
`gemini-3.1-pro-preview`. All retain the pilot's low reasoning setting and
4,096 output-token limit, original system/user messages, and sampling settings.
The main study uses **Batch on all three providers**. OpenAI's synchronous
`service_tier=default` field is omitted because Batch controls scheduling.
Each actual payload and the main config have separate hashes; synchronous
pilot request IDs are not reused. Returned versions and dates must be reported.

The judge remains **`meta-llama/Llama-3.1-8B-Instruct`**, with the existing
rubrics and adjudication. This launch performs generation only. Empty provider
blocks are transport outcomes, not automatically textual refusals or six-policy
labels. Judging and human validation remain separate work.

## Cost review and fixed limits

| Measure | USD |
|---|---:|
| OpenAI planning estimate at Batch rates | 14.95 |
| Anthropic planning estimate at Batch rates | 18.08 |
| Gemini planning estimate at Batch rates | 16.38 |
| Total planning estimate | 49.41 |
| With 50% planning headroom | 74.12 |
| Main generation accounting cap | **160.00** |
| Judge allocation, preserved | 30.00 |
| Setup/contingency allocation, including prior pilot | 60.00 |
| Overall study ceiling | **250.00** |

Estimates reweight the disjoint pilot's costs to main stratum quotas, then apply
the documented 50% Batch discount. They are **not a price guarantee or confidence
bound**: five pilot conditions may not predict all 73. OpenAI's estimate assumes
an explicit HTTP 400 `cyber_policy` rejection with no generated response is
unbilled under Batch's completed-work billing rule. This is an interpretation
of the general Batch policy, not a measured invoice for this exact error code.
The runner records that accounting basis; reconcile provider invoices.

References: [OpenAI Batch](https://developers.openai.com/api/docs/guides/batch),
[OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Anthropic Batch](https://platform.claude.com/docs/en/build-with-claude/batch-processing),
[Gemini Batch](https://ai.google.dev/gemini-api/docs/batch-api), and
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-pro-preview).
These document Batch pricing; Anthropic explicitly lists errored, canceled,
and expired requests as unbilled.

The previous pilot's **$9.402844** conservative accounting total remains intact,
including every old unknown-cost reservation and its $1 infrastructure reserve.
Main Runpod infrastructure has a separate **$5 reserve** within setup/contingency.
No pilot reservation is released by changing transport for the new study.
Prices and software accounting do not control unrelated account spending or
guarantee the provider's eventual invoice.

## Submission and recovery

`configs/frontier-main.json` freezes the budget, transport, settings and hashes.
The separate main ledger never overwrites the pilot. Before each submission it
reserves 32,768 input tokens plus 4,096 output tokens per case at conservative
Batch rates, counting every active batch and unresolved attempt against $160.
The entire 10,950-case maximum-token exposure is **not** submitted at once.

The first wave contains five already-frozen main cases per provider (15 total),
with maximum API reservation **$2.43712**. These are part of the main study,
not extra pilot requests. All three must complete before bulk submission. The
same deterministic case order then continues in batches of at most 200 per
provider, one active batch per provider. This scheduling gate checks Batch API
compatibility and accounting; it does not select items or tune prompts based
on model behavior. Batch turnaround can be up to 24 hours per wave.

Each submission is recorded durably before the network call. An ambiguous POST,
validation/server error, missing/duplicate result identity, unexpected usage,
or unexplained empty answer stops new submissions for review. Explicit terminal
Gemini prompt blocks with missing output usage count as completed while retaining
their full reservation; no zero usage is invented and they are never resent. Completed cases
and explicit safety blocks are never automatically resent. Failed read-only
polls may be retried. Results join by custom ID, never their returned order.
Final results, raw provider records, SQLite backups and status are checkpointed.

Usage-backed responses release unused reservations. Known Batch rejections use
the accounting rules above. Unknown charges retain their full reservation.
Budget exhaustion can leave an incomplete experiment; do not lower token limits,
drop items, change models, or increase the cap to hide an incomplete run.

```bash
# Offline design validation; no credentials or inference calls.
python -m frontier.main_run

# Paid work: poll existing batches and submit one bounded wave.
python -m frontier.main_run --step

# Bounded worker session, with regular checkpoints.
python -m frontier.main_run --watch-minutes 120

# Recover existing remote jobs without submitting new requests.
python -m frontier.main_run --poll-only
```

Default output is the ignored `.local/frontier/main-20261008/` directory.
Resume only its existing ledger; preserve ambiguous submissions for explicit
reconciliation rather than making a new ledger. Provider Batch jobs can continue
after the Runpod CPU worker is deleted. Recover their recorded IDs and results
before scheduling more work. Copy checkpoints before the worker cleanup deadline.
The Runpod control key stays on the local host; the worker needs only the three
provider keys and Anthropic workspace routing. Its local deletion watchdog
depends on the host remaining awake and connected.

## Verification before launch

78 focused tests passed, covering both pilot regressions and main budgets,
preserved payloads, atomic reservations, ambiguous submissions, initial-wave
gating, out-of-order results, duplicate/foreign/missing IDs, usage accounting,
Gemini REST result shape, checkpoints and read-only recovery. The offline main
dry run verifies 10,950 unique IDs and 3,650 cases per provider.

Manifest SHA256: `3070e5cb5923cbe7d23c43b84bb61edf14090131544813633d9cf245a4873fa3`.
Main config SHA256: `0a167c997da0d7483aeb7b11e30bc1788ada85e4692dc9d807090d4227882963`.

## Bounded continuation after launch

The Batch supervisor (`frontier.main_watch`) waits for the first two-hour session
to exit, then resumes the same ledger only after a clean exit. Completion, budget
stops and errors terminate continuation. It never reruns completed cases.
The worker cleanup deadline is 48 hours from creation, with maximum compute of
$2.88 at $0.06/hour within the unchanged $5 infrastructure reserve. The local
watchdog's `--main-study` mode checks this price, reserve and lifetime. It does
not extend the pilot watchdog limit. Backups and API-driven deletion require
the local host to remain awake and connected. This is not a provider-side timer.
