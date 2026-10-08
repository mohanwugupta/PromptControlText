# Revision status — October 7, 2026, Pacific

**The frontier experiment is prepared, and the OpenAI pilot is complete.
Anthropic remains blocked by API billing. The main experiment, new judging,
human validation, and revised analysis are not complete.**

This report combines the user's task list, supplied related-work draft and
revision mapping, repository documentation, and verified pilot records. It
distinguishes existing repository work from work executed in this chat. Live
cluster job status has not been checked here.

| Task | Verified status | What remains / completion criterion |
|---|---|---|
| Clean up GitHub | Marked complete in the user's checklist. Experiment code and records are on `codex/frontier-credential-setup`, draft PR #3. | Review and merge the final PR when ready; keep run provenance and results linked. Pushing a branch is not merging it. |
| Dashboard for rating LLM judges | Existing Streamlit dashboard, blinding, autosave, export, and audit builders are in `audit/`. No new dashboard revision or completed human audit is established by this work. | Align the interface with the canonical six-policy rubric; include the context needed by each separate annotation task; finalize instructions and adjudication; verify label exports and completion tracking. |
| Human-audit sampling | Existing sampling code is present. The frontier generation sample is frozen, but that does not finalize the human-audit sample. | Freeze an audit design covering models, benchmarks, common and rare labels, disagreements, and safety boundaries. Keep a representative random component separate from targeted error discovery; obtain independent human labels and report class-wise errors and agreement. |
| New open-weight models / DeepSeek | Repository README and job config identify four completed prompted snapshots (Gemma 12B, Gemma 31B, Qwen 3.6, Nemotron Nano), and two DeepSeek runs needing completion. This is repository-reported status, not a fresh cluster verification. | Check actual row coverage, nonempty outputs, duplicates, and job logs; finish the two DeepSeek generations; verify the documented CUDA mitigation on the cluster; run the original judge on completed inputs. |
| Frontier models — Sandy | Context repairs, sampling, adapters, durable budget ledger, controlled resume, and Runpod deployment are implemented. OpenAI completed 50 pilot cases. Anthropic has four billing rejections; Google has one quota rejection and is now deferred. | Fix Anthropic billing/key organization, finish its 50 pilot cases, review output quality and costs, then prepare the main-run configuration. Restore Google only after billing is ready; it remains in the frozen three-provider design. |
| Preserve the model rater | Canonical judge confirmed as `meta-llama/Llama-3.1-8B-Instruct`; existing A/B/C rubrics and adjudication retained in the protocol. HF access was verified earlier. No new judge job was run here. | Run that checkpoint and the same rubrics on the new valid responses using Mohan's cluster or a GPU pod; map labels to every request, including repeated response text; human-validate the labels. Report provider blocks and errors separately from textual refusals. |
| Change analysis to d-prime / SDT | Not implemented in the current revision. The original analysis notebook remains. | Define the positive class and endpoint labels first; implement d-prime and criterion with a declared correction for zero/one rates; estimate uncertainty by item/parent; regenerate figures and tables. Retain six-policy transition analysis as a separate outcome. |
| Rewrite related work — Sandy | The revised draft was supplied by the user and appears in the local PDF material. It is a draft, not a citation-verified manuscript integration. | Check primary sources and citation keys; qualify broad claims about prior work; make the within-item intervention, six-policy outcomes, and safety-tradeoff contribution precise; integrate and compile in the manuscript. No editable `.tex` or `.bib` source was found in the supplied folder. |
| Finish the revision | Revision mapping identifies practical implications, taxonomy/judge validation, and generalizability as remaining concerns. | Integrate new evidence into methods/results/discussion; explain operating-point choices and limitations; address sensitivity to policy definitions; update the reviewer-response mapping and reproducibility package. |

## What the frontier preparation established

- Restored required IHEval task system context and HarmBench contextual passages
  for the new extension while preserving historical data and runners.
- Froze 50 stratified main items and 10 disjoint pilot items. All models receive
  identical items. The main study retains all 73 conditions.
- The original three-provider plan has 10,950 main cases and 150 pilot cases.
  The current instruction is OpenAI plus Anthropic for now: 100 pilot cases;
  Google's 50 are deferred. The main design has not silently been reduced.
- Kept model settings and prompts unchanged across pilot attempts. Added
  provider-subset continuation and explicit attempt history without erasing costs.
- Passed 36 focused tests. Preserved the local SQLite ledger, published response
  records, and verified retrieval before deleting all paid workers.

## Current pilot and budget

There are **55 API attempts and 50 completed cases**: 40 OpenAI text responses,
10 OpenAI HTTP safety blocks, four Anthropic billing failures, and one Google
quota failure. The OpenAI cases contain no recorded output-token truncation.
Anthropic's last two retries were separately authorized local one-case billing
checks; all other live pilot attempts ran on Runpod. No main generation or new
judge run has begun.

The original $10 cap is unchanged. Current API accounting is **$7.505053**,
including **$7.241728 of retained reservations** and **$0.263325 of token-based
estimates**. Including the full $1 infrastructure reserve gives **$8.505053**;
approximately **$1.49** remains under that conservative accounting. Estimated
compute across the three deleted workers is **$0.035314**. These figures are
not an invoice. Missing usage and incomplete provider pilots prevent a reliable
full-study cost estimate. The broader $250 study ceiling and its allocations are
planning constraints, not measured expected costs.

The latest evidence is in
[the billing-check record](../artifacts/frontier/runs/2026-10-08-anthropic-billing-check/README.md),
following [the Runpod continuation](../artifacts/frontier/runs/2026-10-08-anthropic-resume/README.md)
and [the initial pilot](../artifacts/frontier/runs/2026-10-08-pilot/README.md).

## Order of work to finish

1. **Resolve the current account blocker.** Confirm that available Anthropic API
   credits and the saved project key belong to the same Console organization.
   Update only the local key if needed. If they already match, use the latest
   request ID with Anthropic support. Resume only after an account update or
   explicit instruction, retaining the same ledger and remaining cap.
2. **Complete and review the pilot.** Finish Anthropic, reconcile reservations,
   assess blocks/truncations and cost by stratum, and decide when Google rejoins.
   Freeze any justified setting changes before main generation.
3. **Execute the main extension.** Use the same frozen items under all 73
   conditions, enforce the reviewed study budget, checkpoint, verify coverage,
   retrieve outputs, and terminate resources. Do not resample after viewing
   comparative model results.
4. **Finish judging and human validation.** Preserve Llama-3.1-8B and its rubrics;
   complete the dashboard and independent audit sample; report confusion,
   class-wise reliability, adjudication, and how label errors affect conclusions.
5. **Complete SDT and policy-transition analysis.** Separate endpoint correctness
   from the output-only policy taxonomy, account for the paired item design,
   and present the small sample as an exploratory generalization check.
6. **Finish the manuscript and reproducibility record.** Sandy can finalize
   related work while billing/model work proceeds. Integrate results, practical
   implications, limitations, and reviewer responses; verify citations and
   compiled manuscript; publish final artifacts and review/merge the PR.

Dashboard preparation, related-work verification, and checking cluster outputs
can proceed while API billing is blocked. Formal notation should clarify the
measured behavior and assumptions; it should not imply a demonstrated internal
mechanism or causal process that this behavioral design does not establish.
