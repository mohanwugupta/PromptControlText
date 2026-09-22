# Frontier API pilot report

This is a technical feasibility and cost pilot, not a safety-performance result.

Source repository commit: `55358c02eb1e8988c93fc4090da6db7e924a3d0f`. All recorded source hashes verified.

Planned: 192 requests on 24 items (8 per benchmark), two models, four conditions per item.
The four conditions are an omitted-system baseline and three assigned prompts. The sparse design covers the 72 registry prompts once per model before exclusions.

Estimated generation charges from returned token usage: **$1.0592**.
Conservative budget accounting including unresolved/rejected-request reservations: **$2.4224 / $10**.
These are local estimates, not a provider invoice. No paid judge calls are included.

## Response quality

| Model | Completed | Incomplete | Empty returned finals | API blocks | Other failures | Skipped variants | Not attempted | Estimated usage cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.5-2026-04-23 | 85 | 0 | 0 | 2 | 4 | 5 | 0 | $0.5352 |
| gpt-6-astra | 83 | 0 | 0 | 3 | 4 | 6 | 0 | $0.5239 |

Both models used low reasoning effort and a 2,048-token combined reasoning/output cap. No temperature or seed was supplied. This differs from the original non-reasoning 512-token protocol.

## Costs by benchmark

| Model | Benchmark | Returned responses | Input tokens | Output tokens | Reasoning tokens | Mean cost/response | Mean latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| gpt-5.5-2026-04-23 | HarmBench | 23 | 1041 | 8283 | 1251 | $0.01103 | 13.7 |
| gpt-5.5-2026-04-23 | XSTest | 31 | 1118 | 6631 | 898 | $0.00660 | 11.2 |
| gpt-5.5-2026-04-23 | IHEval | 31 | 1558 | 2308 | 1184 | $0.00248 | 8.1 |
| gpt-6-astra | HarmBench | 22 | 1017 | 4143 | 608 | $0.00988 | 16.0 |
| gpt-6-astra | XSTest | 31 | 1118 | 4397 | 330 | $0.00745 | 14.1 |
| gpt-6-astra | IHEval | 30 | 1534 | 1205 | 264 | $0.00252 | 10.7 |

## Preliminary generation-only projections

Extrapolations use each model’s observed pooled mean. Item selection, missing API-blocked cases, truncation, output lengths, and sparse prompt assignment limit these estimates. They are not guaranteed quotes.

| Model | 100 items × 25 conditions | 200 items × 25 conditions | 100 items × 73 conditions | 600 items × 73 conditions |
|---|---:|---:|---:|---:|
| gpt-5.5-2026-04-23 | $15.74 | $31.48 | $45.97 | $275.81 |
| gpt-6-astra | $15.78 | $31.56 | $46.08 | $276.49 |

The 25-condition option retains eight families × three clarity levels plus baseline, with one preselected wording variant per cell. It sacrifices paraphrase replication and must be reported as a reduced extension. A full budget must also cover judging, failed requests, reruns, and contingency.

## Interpretation limits and next steps

- Provider policy errors are API-level blocks, not generated refusals. The affected item’s remaining variants are skipped across both models; no prompt rewriting is used to evade the block.
- Prior credit-exhaustion attempts are retained in the attempt history but do not count as generated responses.
- Inspect incomplete and mixed responses before selecting a production output cap. Increasing the cap can increase cost.
- No LLM or human labels have been assigned; these data cannot yet support policy-switching or SDT estimates.
- A limited assistant spot-check of four benign XSTest baseline outputs (two items, both models) found relevant final answers. This is not a blinded human audit or an accuracy estimate.
- HarmBench contextual inputs omit ContextString, matching the existing loader. IHEval inputs are flattened and may omit the native instruction hierarchy. Resolve these dataset issues before claiming benchmark validity.
- Sparse assignments confound item and prompt condition. Use a fully crossed selected subset for any confirmatory extension.
- Keep API policy blocks and missing data in reports; do not silently convert them to refusal or discard them from denominators.
- Reuse original-model outputs only where item text, prompt conditions, and scoring definitions match.

## Budget recommendation

For the $250 total budget, first consider 100 stratified items × all 73 conditions × both models. This preserves wording replication and leaves room for judging, reruns, and contingency. The pilot suggests roughly $92 for generation alone at standard prices, but nonrandom API exclusions and short flattened IHEval inputs may underestimate production costs. Fix the dataset construction and scoring protocol before a larger run. No production run has been started.

## Files

- `manifest.json`: exact planned requests, sampling, provenance, prices, and settings.
- `usage.jsonl`: public attempt metadata and token usage, excluding output text and provider identifiers.
- `responses.jsonl`: private append-only request history and final texts (gitignored).
- `pilot_results.csv`: one row per planned task, including missing and blocked rows.
- `ledger.jsonl`: private persistent reservations and settlements (gitignored); `budget_accounting.json` publishes the aggregate.
