# Anthropic continuation on Runpod

This continuation ran on October 7, 2026 Pacific (October 8 UTC). After the user
reported adding Anthropic API credits, the prior credit rejection was archived
with its full cost reservation. The unchanged pilot case was retried once.
Anthropic again returned HTTP 400: **Your credit balance is too low**. No new
text responses were produced, and the other 49 Anthropic cases were not attempted.

The first CPU worker had an unreachable SSH port, and Runpod's restart operation
also failed. It was deleted without receiving credentials or experiment data.
A replacement CPU worker passed its dry run. Following explicit user approval,
only the Anthropic key was transferred over SSH to a root-only file. A private
comparison confirmed that the worker used the project's saved key with no
environment override. The failed attempt was retrieved and verified before the
worker was deleted. Both deletions were confirmed with HTTP 404 and pod-list
absence. Neither worker created a network volume.

At this checkpoint, the original pilot has **53 attempts and 50 completed cases**:
40 OpenAI text responses, 10 OpenAI provider safety blocks, two Anthropic billing
rejections, and one Google quota rejection. All original OpenAI records are
unchanged. Google remains deferred at the user's request; no Google calls were
made in this continuation.

The cumulative API ledger is **$7.013533**, including $6.750208 in reservations
for missing usage. Including the full original $1 infrastructure reserve gives
**$8.013533 against the $10 cap**. The two workers added approximately $0.0182
in compute. These are accounting estimates and reservations, not confirmed bills.

See [run.json](run.json), [infrastructure.json](infrastructure.json),
[attempts.jsonl](attempts.jsonl), [report.json](report.json),
[reconciliation.json](reconciliation.json), and [provenance.json](provenance.json).
The full local export matches the downloaded SQLite checkpoint. Prior failed
attempts are embedded in its `prior_attempts` and still count against the cap.
The public attempt record references unchanged inputs by request/item/condition
IDs; original OpenAI outputs remain in the initial pilot record. The complete
resumable ledger and joined export are preserved locally, outside Git. All 36
focused tests passed.

A subsequent local billing check is recorded in
[the next checkpoint](../2026-10-08-anthropic-billing-check/README.md).
