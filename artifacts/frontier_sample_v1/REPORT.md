# Astra corrected-sample progress

Planned: 7300; submitted: 4984; returned: 4063.
Completed nonempty responses ready for original judging: 4045.
Usage-based cost estimate: $10.869015. Conservative pending/error accounting: $74.937969 / $75.

## Planned-row status

| Status | Rows |
|---|---:|
| api_error | 18 |
| completed | 4045 |
| not_submitted | 1843 |
| pending_batch | 921 |
| skipped_provider_block | 473 |

## Batch status

| Batch | State | Reconciled |
|---|---|---|
| 1 | completed | True |
| 2 | completed | True |
| 3 | upload_interrupted_no_inference | True |
| 4 | completed | True |
| 5 | completed | True |
| 6 | completed | True |
| 7 | in_progress | False |

## Interpretation and handoff

No production judge has run. Use the original Llama-3.1-8B pipeline on the collaborator cluster; confirm the checkpoint from the original manifests.
Public metadata excludes raw responses. judge_input.private.csv contains completed outputs; all_statuses.private.csv retains every planned row.
The corrected sample is 30 HarmBench, 40 XSTest, and 30 native IHEval system-prompt-extraction items, with all 73 controller conditions.
Native IHEval system instructions remain in the no-controller baseline. Corrected inputs require matched historical-model reruns.
Provider bio/cyber policy errors are not model refusals; later unsubmitted variants of affected items are skipped. No replacements or prompt rewriting.
Costs are estimates, not invoices. Input is priced as uncached; possible cache-write surcharges are not itemized in returned usage. Pending reservations and settled budget accounting include a 25% input margin.
