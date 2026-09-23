# Astra corrected-sample progress

Planned: 7300; submitted: 1100; returned: 100.
Completed nonempty responses ready for original judging: 95.
Usage-based cost estimate: $0.260810. Conservative pending/error accounting: $68.344069 / $75.

## Planned-row status

| Status | Rows |
|---|---:|
| api_error | 5 |
| completed | 95 |
| not_submitted | 5840 |
| pending_batch | 1000 |
| skipped_provider_block | 360 |

## Batch status

| Batch | State | Reconciled |
|---|---|---|
| 1 | completed | True |
| 2 | in_progress | False |

## Interpretation and handoff

No production judge has run. Use the original Llama-3.1-8B pipeline on the collaborator cluster; confirm the checkpoint from the original manifests.
Public metadata excludes raw responses. judge_input.private.csv contains completed outputs; all_statuses.private.csv retains every planned row.
The corrected sample is 30 HarmBench, 40 XSTest, and 30 native IHEval system-prompt-extraction items, with all 73 controller conditions.
Native IHEval system instructions remain in the no-controller baseline. Corrected inputs require matched historical-model reruns.
Provider bio/cyber policy errors are not model refusals; later unsubmitted variants of affected items are skipped. No replacements or prompt rewriting.
Costs are estimates, not invoices. Input is priced as uncached; possible cache-write surcharges are not itemized in returned usage. Pending reservations and settled budget accounting include a 25% input margin.
