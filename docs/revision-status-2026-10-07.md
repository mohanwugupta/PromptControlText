# Revision status and remaining checklist — October 7, 2026, Pacific

**The three-provider frontier pilot is complete: 150/150 cases. The main
experiment, new judging, human validation, SDT analysis, and final manuscript
revision are not complete.** This report follows the original task list and
focuses on Sandy's frontier-model and related-work responsibilities.

**Main-run update:** generation stopped for accounting review after four Gemini
provider blocks omitted output-token usage. The worker was deleted after a
verified backup. Read-only recovery now accounts for 1,411 completed cases,
4 review cases, 200 pending submitted cases, and 9,335
unsubmitted cases. See the [stop report](../artifacts/frontier/runs/2026-10-08-main/STOP-REPORT.md).
The frozen design and budgets are unchanged. PR #3 was merged; later recovery
records are being published in a follow-up PR. Generation has no reliable ETA
until the accounting handling is reviewed and execution resumes.

## What you originally asked for

Review the supplied folder and connected PromptControlText repository; use the
revision agreement and supplied related-work rewrite to plan the remaining work;
run the frontier extension through Runpod; and push code and run records to
GitHub. The working decisions are:

- One generator per company: OpenAI, Anthropic, and Google, superseding the earlier
  six-model suggestion.
- Preserve the documented **`meta-llama/Llama-3.1-8B-Instruct`** judge and its
  original A/B/C rubrics and adjudication. You explicitly confirmed 8B rather
  than the tentative 70B recollection. No judge substitution was made.
- Keep the same items across providers and all 73 conditions in the main study.
- Run a 150-case pilot within the authorized **$10 total cap** before main work.
- Keep credentials private and publish reproducible experiment records.

The report draws on the supplied paper/revision material, your pasted draft,
repository files, and actual run records. Existing repository work is separated
from work performed here. Live cluster/DeepSeek completion has not been verified.

## Original checklist: current status

| Original task | Status | What remains before it is complete |
|---|---|---|
| Clean up GitHub | **Marked done in your original checklist.** PR #3 containing pilot/main-launch work is merged. Recovery records continue on `codex/frontier-credential-setup`. | Review the follow-up recovery/results PR when ready; later pushes are not automatically part of merged PR #3. |
| Create/edit dashboard to rate LLM judges | **Existing dashboard; revision not completed here.** `audit/` already contains a Streamlit interface, blinding, autosave, tracking, exports, and audit builders. | Review the interface against the six-policy rubric, improve instructions and adjudication, verify exports, and run the human audit. Preserve output-only policy labeling; any context-rich endpoint assessment must be a separate annotation task. |
| Figure out sampling | **Frontier generation sampling done; human-audit sampling pending.** These are different samples. | Freeze the human-audit design, annotator allocation, and adjudication rules. Include a representative random component for error estimates and a separately reported targeted component for rare labels/disagreements. |
| Get new models running / check DeepSeek | **Partially documented; live completion unverified.** Your checklist marked the DeepSeek check done. The repository reports four completed prompted snapshots and two DeepSeek runs still needing completion. | Verify current cluster outputs and logs; finish missing generations if still necessary; check coverage, duplicates, empty/truncated outputs, and judge all valid new responses. The original generic “Check” subitem has no specific completion criterion yet. |
| Change to d-prime / SDT | **Not implemented in this revision.** | Specify positive class and endpoint labels, treatment of blocks/missing data, extreme-rate correction, d-prime and criterion, and paired/item-clustered uncertainty. Regenerate figures and tables. Keep six-policy transitions as a separate analysis. |
| Run frontier models — Sandy | **Pilot complete. Main extension started, then stopped for accounting review.** | Resolve the four terminal Gemini blocks without resending them, recover pending batches, resume the existing frozen ledger, then judge and analyze all accounted cases. Pilot execution alone does not establish whether the paper's pattern generalizes. |
| Rewrite related work — Sandy | **Your revised draft exists; final verification/integration pending.** | Check primary sources and citation keys, qualify broad claims about prior work, sharpen the within-item/policy-transition contribution, then integrate and compile the manuscript. No editable `.tex` or `.bib` files were found in the supplied folder. |

The four repository-reported completed snapshots are Gemma 12B, Gemma 31B,
Qwen 3.6, and Nemotron Nano. The two outstanding repository entries are
DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Llama-70B. Each prompted run
expects 242,640 unique, nonempty rows (`3,370 × 72`). These are not newly verified
cluster results. The documented 512-token cap and CUDA mitigation also need
checking against actual logs before treating those runs as ready for analysis.

## Verified work completed here

- [x] Reviewed the supplied research/revision context and relevant repository pipeline files.
- [x] Set up local, Git-ignored credentials and verify live access to all three generators.
- [x] Resolve Anthropic account/workspace routing and Gemini's earlier quota blocker.
- [x] Restore missing IHEval task-system context and HarmBench contextual passages for the new extension, preserving historical data/runners.
- [x] Freeze 50 main items and 10 disjoint pilot items, stratified across the three benchmarks.
- [x] Preserve all 73 main conditions: `50 items × 73 conditions × 3 providers = 10,950 cases`.
- [x] Freeze the pilot: `10 items × 5 conditions × 3 providers = 150 cases`.
- [x] Implement provider adapters, durable checkpoints, duplicate protection, shared cost reservations, and controlled recovery from known setup errors.
- [x] Complete every pilot case without rephrasing or resending provider-blocked prompts or using fallback models.
- [x] Preserve every failed setup attempt and its reservation; correct one recorded Anthropic refusal classification without a new API call.
- [x] Pass 56 focused tests; verify all case IDs/content against the frozen design and SQLite against JSONL.
- [x] Retrieve and preserve results locally; delete all four pilot workers and stop the final watchdog/sleep inhibitor. Unrelated account resources were left alone.
- [x] Record code, outcome/usage metadata, response hashes, verification evidence, and this checklist in the GitHub branch/PR, excluding credentials and private account/workspace IDs.
- [x] Publish the 99 complete new response records (49 Anthropic and 50 Gemini) after explicit user approval. Verified every response against its previously published text hash and the local ledger; credentials remain excluded.

## Pilot outcome and budget

| Generator | Cases completed | Text responses | Provider blocks | Recorded truncations |
|---|---:|---:|---:|---:|
| OpenAI `gpt-6-astra` | 50 | 40 | 10 | 0 |
| Anthropic `claude-opus-5-5` | 50 | 39 | 11 | 0 |
| Google `gemini-3.1-pro-preview` | 50 | 50 | 0 | 0 |
| **Total** | **150** | **129** | **21** | **0** |

These counts describe API/output outcomes, not the six behavioral labels or
safety rankings. A text response can itself be a refusal. Empty provider blocks
need separate reporting and cannot simply be sent to the text judge as refusals.
There were **156 API attempts**: the 150 final case outcomes plus six preserved
setup failures. Most calls ran on Runpod; four one-case Anthropic setup checks
ran locally, as recorded. Main generation has since started; no new judge run has started.

| Budget measure | USD | Interpretation |
|---|---:|---|
| Token usage at configured conservative rates | 0.915356 | Estimate from returned usage, not a reconciled invoice; some blocks may be unbilled. |
| Retained reservations | 7.487488 | Possible charges held for missing usage and prior failures, not confirmed spending. |
| Total API ledger | 8.402844 | Sum of usage estimates and retained reservations. |
| Full infrastructure reserve | 1.000000 | Kept intact for enforcing the pilot cap. |
| **Conservative total against the $10 cap** | **9.402844** | **Pilot remained under the cap.** |
| Estimated compute across all four pilot workers | 0.061027 | Elapsed time at recorded compute rates; not exact invoices/storage charges. |

No earlier reservation was silently released. The Anthropic and Gemini pilot
costs support preliminary main-run projections, but OpenAI's retained block
reservations prevent a reliable combined synchronous cost estimate. The separate
main Batch plan documents its billing assumptions and preserves all old
reservations without releasing them. The broader $250 study ceiling and its proposed
allocations remain planning constraints, not a confirmed final experiment price.

Evidence: [final pilot record](../artifacts/frontier/runs/2026-10-08-gemini-completion/README.md),
[coverage/quality summary](../artifacts/frontier/runs/2026-10-08-gemini-completion/quality-summary.json),
[checkpoint verification](../artifacts/frontier/runs/2026-10-08-gemini-completion/verification.json),
and [worker cleanup](../artifacts/frontier/runs/2026-10-08-gemini-completion/cleanup.json).

## Remaining steps, in order

1. **Review and close out the pilot.** Reconcile costs against provider billing;
   inspect response completeness and context fidelity; declare how provider
   blocks enter endpoint analyses and denominators. The 21 blocks are part of
   the observed outcome, not errors to evade. Preserve the frozen sample.
2. **Resolve the main-run stop and prepare judge validation.** The main config,
   Batch transport, and hard budgets are frozen. Recover existing batches and
   review the four terminal blocks while retaining unknown-cost reservations.
   Validate the accounting handling before resuming the same ledger. Run the original
   8B judge/rubrics on the 129 pilot text responses and check row-level mapping,
   including identical response text. This judging has not happened yet.
3. **Finish dashboard and human-audit design.** Set the human rubric, independent
   rater allocation, representative/targeted sampling, blinding, and adjudication.
   Freeze the design before using comparative model results to select cases.
4. **Run the main frontier extension.** Execute all 10,950 frozen cases, verify
   coverage and returned model versions, preserve failures/blocks, retrieve
   results, and terminate resources. The 50 independent items support an
   exploratory extension, not precise benchmark-wide or fine-subgroup rankings.
5. **Complete open-weight generation and judging.** Verify the four reported
   snapshots and two DeepSeek jobs; finish anything missing. Judge the new valid
   outputs with the same 8B checkpoint and rubrics, and map every label back to its
   generation request. Mohan's cluster is an available proposed route, not a job
   already submitted by this work.
6. **Complete human validation and analysis.** Collect independent human labels,
   assess class-wise judge errors/agreement, adjudicate disagreements, and report
   how label uncertainty affects conclusions. Implement SDT/d-prime and criterion
   with declared corrections and paired uncertainty; analyze six-policy
   transitions separately from harmful-compliance/false-refusal endpoints.
7. **Finish the manuscript and reviewer response.** Verify and integrate Sandy's
   related-work draft; update methods, results, practical implications, taxonomy
   sensitivity, limitations, and generalizability. Map each reviewer concern to
   a concrete change, compile/check the paper and references, and finalize the
   reproducibility package and PR.

Related-work verification, dashboard preparation, and checking cluster outputs
can proceed in parallel with main-run planning. The revision mapping also asks
for clearer practical implications and sensitivity to the manually designed
policy taxonomy; those remain open beyond the short original checklist.

## Inputs still needed

- **Editable manuscript and bibliography location** (`.tex`/`.bib`, Overleaf source,
  or the actual authoring project) to integrate the draft and compile the paper.
- **Current open-weight/DeepSeek outputs and job logs, or scoped cluster access**
  to verify what has finished rather than relying on repository notes.
- **Provider billing reconciliation** for uncertain reservations. A main-run
  budget/configuration is now prepared and preserves all old reservations.
  No additional provider credentials are missing now.
- **Human annotation arrangements:** who will annotate, available effort, and who
  adjudicates. The exact audit size and agreement criteria still need definition.

The framing should remain behavioral: explicit prompt interventions, observed
response policies, and within-item safety tradeoffs. Formal notation should
clarify the process and assumptions without implying a demonstrated internal
mechanism. The supplied related-work draft already points in this direction,
but its citations and broad comparisons have not yet been independently verified.
