# Runpod pilot: October 8, 2026 UTC

**Status: partial; blocked on Anthropic and Google API billing.** The frozen
150-case pilot ran on a Runpod CPU worker. All 50 OpenAI cases completed: 40
returned text and 10 returned HTTP `cyber_policy` blocks. These blocks are
provider outcomes, not textual refusals. No blocked prompt was retried or
rephrased. Anthropic and Google each rejected their first call for billing or
quota reasons. In total, 52 API attempts produced 50 completed cases, two
billing errors, and 98 unattempted cases.

| Provider | Model | Completed / planned | Result |
|---|---|---:|---|
| OpenAI | `gpt-6-astra` | 50 / 50 | 40 text responses; 10 provider blocks |
| Anthropic | `claude-opus-5-5` | 0 / 50 | HTTP 400: API credit balance too low |
| Google | `gemini-3.1-pro-preview` | 0 / 50 | HTTP 429: Pro free-tier quota is zero |

The worker was deleted at **01:55:50 UTC** (October 7, 18:55 Pacific). A subsequent
GET returned 404, and the pod was absent from the account's pod list. No network
volume was created. The local deletion watchdog and keep-awake process were
stopped after cleanup was verified.

## Budget

The authorized cap was **$10 total**, divided into $9 for API accounting and $1
reserved for Runpod infrastructure. The ledger remained below that cap:

| Component | USD |
|---|---:|
| Returned API token usage at conservative rates | 0.263325 |
| Reservations retained for 12 calls without usage | 6.504448 |
| Total API accounting | 6.767773 |
| Estimated Runpod compute at $0.06/hour | 0.017114 |
| Conservative total using the full $1 infrastructure reserve | 7.767773 |

These are accounting figures, not a confirmed invoice. The reservations are
possible charges held against the cap, not measured spending. Actual compute is
estimated from pod lifetime; the infrastructure reserve also covers storage and
billing settlement. Missing usage and the incomplete three-provider pilot prevent
a reliable full-study cost estimate. `projected_main_accounting_usd_by_provider`
in the report includes reservations and must not be interpreted as expected spend.

## Record and validation

- [run.json](run.json): status, counts, budget, blockers, and verified cleanup.
- [results.jsonl](results.jsonl): all 52 request records, outputs, usage, and errors.
- [report.json](report.json): reproducible accounting and completion summary.
- [config.json](config.json): exact pilot configuration snapshot.
- [infrastructure.json](infrastructure.json): worker specification and lifetime.
- [provenance.json](provenance.json): code hashes and retrieved archive checksum.
- [progress.jsonl](progress.jsonl): execution events, including partial stops.
- [reconciliation.json](reconciliation.json): an existing OpenAI HTTP block was
  reclassified without replay; its full reservation was retained.

The sample and prompts remain those in the frozen
[manifest](../../manifest.json). Manifest and configuration content hashes are
in `run.json`; raw file checksums are recorded separately in `provenance.json`.
Provider-subset continuation and HTTP-block bookkeeping were added during the
pilot; model settings, prompts, and sampled items were unchanged. The report's
expected-spend projection was subsequently tightened to withhold estimates when
usage is missing.

All 31 focused tests passed. Exported rows match the retrieved SQLite checkpoint,
have unique IDs from the frozen design, and contain no unresolved in-flight calls.
The SQLite ledger and credentials remain local and excluded from Git. Main
generation remains disabled, and no judge was run. The canonical judge remains
`meta-llama/Llama-3.1-8B-Instruct`.

## Resume requirements

1. Add API credits to the organization owning the existing Anthropic key in
   [Claude Console billing](https://platform.claude.com/settings/billing).
2. Enable paid billing for the Google project owning the existing Gemini key;
   follow the [Gemini API billing guide](https://ai.google.dev/gemini-api/docs/billing/).
3. Reconcile the two failed attempts with explicit attempt-history records,
   retaining prior costs. Restore `.local/frontier/pilot.sqlite` to a new Runpod
   CPU worker and continue only the unfinished providers within the remaining
   original $10 cap. Do not discard the ledger or repeat OpenAI cases.
4. Retrieve and verify the updated records, terminate the worker, and reconcile
   usage and billing before deciding whether the full experiment is ready.

New credentials are unnecessary if billing is enabled for the existing key
organizations/projects. Funding an account is separate from pilot expenditure.
