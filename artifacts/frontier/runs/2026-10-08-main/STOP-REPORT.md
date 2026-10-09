# Main study stopped for accounting review

Snapshot: 2026-10-08T04:54:40.231324+00:00. Generation stopped at approximately 9:40 PM
Pacific on October 7, 2026. The worker was deleted after a verified final backup;
[cleanup.json](cleanup.json) records the successful deletion checks.

Of 10,950 frozen cases, **1,411 are completed**, **4 require accounting review**,
**200 are submitted but still pending**, and **9,335
have not been submitted**. Completion includes provider blocks and is not a count
of text answers. The three-provider pilot remains complete at 150/150.

| Provider | Completed cases | Review | Pending |
|---|---:|---:|---:|
| OpenAI | 805 | 0 | 0 |
| Anthropic | 405 | 0 | 200 |
| Google | 201 | 4 | 0 |

The four review cases returned Gemini `promptFeedback.blockReason=OTHER` with no
candidates. Their usage lists prompt and total tokens but omits output and
thought token counts. The runner correctly preserved all four full reservations
($0.229376 combined); its unknown-usage gate then stopped new submissions.
These are terminal provider blocks, not requests to retry or rephrase. Exact IDs
and non-text diagnostics are in [stop-report.json](stop-report.json); original
provider rows and normalized records are preserved unchanged.

Read-only recovery used the existing ledger and GET requests to retrieve
already-submitted jobs. No new inference requests were submitted. The final
worker checkpoint and subsequent recovery archives remain locally preserved;
[recovery.json](recovery.json) records the immediate checkpoint ancestry.

Main accounting is **$30.399940**, consisting of
$5.594564 usage estimates,
$0.229376 retained unknown-usage
reservations, and $24.576000 pending reservations.
Including the unchanged prior-pilot accounting and $5 main infrastructure reserve,
the study accounts for **$44.802784**.
These figures are not reconciled invoices. The $160 generation cap, $250 overall
ceiling, and $30 judge allocation remain unchanged. Estimated worker compute was
$0.047836; the full $5 infrastructure reserve is retained.

Next: retrieve any remaining submitted results, review terminal-block accounting
without releasing unknown charges or resending blocked cases, validate the fix,
and resume only unsubmitted frozen cases from this ledger on an authorized worker.
There is no reliable completion ETA while generation is stopped. The scheduled
monitor is being disabled after this stop notification. Judging with the original
Llama-3.1-8B-Instruct, human validation, analysis and manuscript revision are pending.

PR #3 was merged on October 8 at 04:45 UTC. This report and later results belong
to the follow-up commit/PR on `codex/frontier-credential-setup`.
