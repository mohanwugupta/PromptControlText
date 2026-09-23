# Sandy's frontier API extension: scope, budget, and remaining work

Status: feasibility pilot complete; production generation and judging not started.
The pilot returned 168 completions for an estimated $1.0592 of token usage.
Conservative local accounting is $2.4224 including reservations for unresolved
requests; neither number is a provider invoice. Five API policy blocks, eight
transport failures, and eleven skipped variants are recorded separately.
No safety labels or SDT results have been produced.

## What “full run” means

The original design has 3,370 items (400 HarmBench, 450 XSTest, 2,520 IHEval)
and 73 conditions (72 registry prompts plus an omitted-system baseline).
That is 246,010 generation requests per additional model, or 492,020 for both
pilot models. This adds frontier models; it does not regenerate the original
four models. A 600-item experiment is a subset, not the original full dataset.

## Cost scenarios in USD

| Scope | Standard generation only | Batch generation only | Standard generation + assumed judging + 25% buffer | Batch equivalent with buffer |
|---|---:|---:|---:|---:|
| All items, GPT-5.5 | 996–1,549 | 498–775 | 3,016–3,708 | 1,508–1,854 |
| All items, GPT-6 Astra | 997–1,553 | 498–776 | 3,017–3,712 | 1,509–1,856 |
| All items, both models | 1,993–3,102 | 996–1,551 | 6,033–7,420 | 3,017–3,710 |
| 100 items × 73 conditions, both | 92 | 46 | 220 | 110 |

These are planning scenarios, not confidence intervals or guaranteed ceilings.
The lower full-dataset estimate weights each benchmark's observed mean by its
actual item count. The upper scenario applies the pilot's pooled mean to all
items. IHEval dominates the full dataset and was cheap in this pilot, but its
flattened input may omit essential context. Correcting that can increase costs
beyond the displayed scenarios. The 100-item estimate uses the pooled mean;
final stratification, prompt effects, failures, and longer outputs can change it.

Generation uses observed token charges at standard prices: GPT-5.5 $5 input /
$30 output per million tokens; GPT-6 Astra $10 / $50. Input is conservatively
priced as uncached. Pricing sources:
[GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5),
[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).

Judging is a separate, unvalidated planning assumption: GPT-5.4 mini at $0.75
input / $4.50 output per million tokens, 1,200 input and 200 output tokens per
judgment, three prompt-based judgments per response, plus one adjudication on
20% of responses. This costs $0.00576 per generated response: $2,834 for both
full models or $84 for the 100-item extension. Measure actual judge token use
and human agreement before adopting this judge. Keeping the original local
judge instead changes the cost and comparability calculation; GPU costs are
not included here. See [mini pricing](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

The [Batch API](https://developers.openai.com/api/docs/guides/batch) offers 50%
lower token prices with asynchronous processing. Estimates assume model and
endpoint eligibility. The current runner does not implement Batch. Account
limits and queued-token capacity also affect completion time. No human audit
labor, storage, or Runpod charges are included. Forecasts are reproducible with
`python3 analysis/forecast_api_cost.py` from the published usage metadata.

For the existing $250 total ceiling, use a carefully stratified 100-item,
fully crossed extension as the candidate design. Its approximately $220
standard-cost scenario plus the pilot fits the ceiling, but only narrowly.
A corrected-input mini-pilot and measured judge costs must confirm affordability
before production. Reduce sample size or use Batch if needed; never silently
increase the ceiling. The $10 top-up funds the pilot, not the full extension.

## Changes included in this branch

- Standalone API pilot runner, immutable task manifest, source hashes, recorded
  settings, durable reservations, process locking, and conservative retry rules.
- Offline tests for budget/resume controls and a guard against rerunning a
  published archive without its private cost ledger.
- Public usage metadata, aggregate reservation accounting, pilot report, and
  reproducible cost forecasts. Generated text, request IDs, raw ledgers, and API
  credentials stay local and are ignored by Git.
- README links and instructions separating pilot evidence from paper results.

## Required before production (not implemented by this pilot)

1. **Validate benchmark construction.** Audit `benchmarks/download_data.py`
   against native HarmBench contextual inputs and IHEval hierarchy. Version any
   corrected data and loaders, retain legacy snapshots, and identify which old
   model outputs can still be compared. An input change can require matched
   baseline reruns; those costs are outside the API estimates above.
2. **Freeze the scientific design.** Record sample IDs, strata, seed, all 73
   conditions, actual model IDs, reasoning effort, output caps, and a missing-data
   policy. Resolve the 512-token historical versus 2,048-total-token pilot
   protocol difference. The sparse 24-item pilot cannot estimate prompt effects.
3. **Build the production adapter.** Generalize `experiments/run_api_pilot.py`
   into explicit per-run configuration and versioned output paths; add Batch
   submission, polling, reconciliation, and per-request reservations. Reconcile
   unknown requests before retrying. Honor API policy blocks without altering
   prompts to evade them. Enforce the approved total cap across generation and
   judging; a generation-only cap is insufficient.
4. **Validate and integrate scoring.** Adapt `scoring/llm_policy_judge.py` to
   the chosen judge, preserve A/B/C votes and adjudications, and run a blinded,
   stratified human audit including refusals, mixed responses, and disagreements.
   Do not treat API errors as generated refusals. Do not reuse the regex fallback
   as a validated judge for new output styles.
5. **Integrate analysis.** Remove hardcoded four-model assumptions in
   `analysis/analysis.Rmd`. Preserve item pairing, report missingness and
   truncation, and use item-level uncertainty estimates. Define a justified
   binary endpoint for d-prime/criterion on safe versus unsafe items, with an
   explicit zero/one-rate correction. Keep the six behavioral-policy outcomes
   separate; IHEval conflict labels are not safe/unsafe labels.
6. **Update the paper only after validation.** Report the extension's actual
   scope and settings, dataset changes, audit agreement, costs, and limitations.
   Sandy's related-work rewrite is a separate deliverable: `%v2` alone is not
   sufficient while the active document still inputs the old related-work file.
   Reconcile both versions, check novelty claims against instance-level prompt
   sensitivity work, and verify/add relevant citations. The manuscript source
   tree and bibliography are not in this checkout; no paper edits are claimed.

Keep these production changes in subsequent reviewable commits, and publish
validated findings only after the corrected-input and judge checks pass.
