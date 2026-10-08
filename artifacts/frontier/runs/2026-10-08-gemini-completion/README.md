# Completed three-provider frontier pilot

**All 150 frozen pilot cases are complete**, with 50 cases per provider. Gemini's
50 calls ran on the same Runpod worker after explicit key-transfer approval.
The earlier zero-free-tier failure was archived with its full reservation before
one unchanged retry, which succeeded. There were no further Gemini failures.

| Model | Text responses | Provider blocks | Total cases | Recorded truncations |
|---|---:|---:|---:|---:|
| OpenAI `gpt-6-astra` | 40 | 10 | 50 | 0 |
| Anthropic `claude-opus-5-5` | 39 | 11 | 50 | 0 |
| Google `gemini-3.1-pro-preview` | 50 | 0 | 50 | 0 |
| Total | 129 | 21 | 150 | 0 |

These are transport/output outcomes, not judged behavioral policies or safety
scores. A text response can itself contain a refusal. Provider blocks are kept
separate and were never rephrased, resent, or sent to fallback models.
There were 156 API attempts in total: 150 final outcomes and six earlier setup
failures. All prompts, conditions, model settings, and case IDs stayed frozen.

[Verification](verification.json) confirms all 150 unique cases match the frozen
design, the SQLite checkpoint exactly matches its JSONL export, source hashes
match, all earlier OpenAI/Anthropic records are unchanged, and no requests remain
in flight. [Quality summary](quality-summary.json) records model IDs, coverage,
transport outcomes, and truncations. All 56 focused tests pass.

The final API ledger accounts for **$8.402844**, including **$0.915356 in
conservative token-based estimates** and **$7.487488 in retained reservations**.
With the full $1 infrastructure reserve, the conservative total is **$9.402844
against the original $10 cap**. Estimated compute across all four pilot workers
is **$0.061027**. These are not final invoices. Earlier reservations remain;
some classifier refusals may not be charged. Full-study OpenAI cost projections
still require billing reconciliation before a reliable combined budget estimate.

After retrieving and verifying the final checkpoint, the worker was deleted.
The API returned 204 for deletion, 404 for subsequent lookup, and the worker was
absent from the pod list. The local watchdog and sleep inhibitor were stopped.
All four workers created for this pilot are deleted. Unrelated account resources
were left alone. See [cleanup evidence](cleanup.json).

[Attempt records](attempts.jsonl) contain all 50 new Gemini outcomes, including
full generated response text, outcome/usage metadata, and references to the
already-published frozen inputs. The full response records were published after
explicit user approval and verified against the previously published text hashes
and the local ledger. The earlier
quota rejection and its retained cost are in [reconciliation.json](reconciliation.json).
Anthropic's final 49 outcomes are in [its completion record](../2026-10-08-anthropic-completion/README.md),
its first success is in [the workspace check](../2026-10-08-anthropic-workspace-check/README.md),
and OpenAI's outcomes are in [the initial pilot](../2026-10-08-pilot/README.md).
Complete joined exports and the resumable SQLite ledger are saved locally and
ignored by Git. API credentials and workspace/account IDs are not published.

**The pilot is complete; the main study and judging have not run.** Next: review
costs and output quality, validate the original Llama-3.1-8B judge on pilot text,
and freeze a separate main-run configuration. See the
[full revision status and checklist](../../../../docs/revision-status-2026-10-07.md).
