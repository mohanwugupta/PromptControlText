# Anthropic pilot completion on Runpod

**All 50 Anthropic cases are complete: 39 text responses and 11 provider blocks,
with no recorded truncation.** One successful configuration check had already
completed locally; the remaining 49 calls ran on Runpod worker
`j36p1t0z02ydon`. OpenAI remains complete at 40 text responses and 10 HTTP blocks.
The user has re-enabled Gemini, so current progress is **100/150 pilot cases**.

The first new Anthropic call returned an empty response with the documented
`stop_reason=refusal`. The original adapter treated it as unexplained `empty`
and stopped. We preserved that result, corrected it to a terminal provider
block, and continued other cases without resending it. The correction is in
[normalization-correction.json](normalization-correction.json). New refusals
retain `stop_details`; ten captured categories were `cyber`. The first block's
category was not captured by the old adapter and has not been inferred.
No prompts, model settings, or frozen case IDs changed; no fallback model ran.
See [Anthropic's refusal documentation](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback).

There are **106 API attempts and 100 completed cases**. All five earlier
Anthropic setup failures and the original Gemini quota failure remain in the
ledger. API accounting is **$8.075488**, including $0.588000 in conservative
token-based estimates and $7.487488 of retained reservations. The conservative
total including the entire $1 infrastructure reserve is **$9.075488 / $10**.
About $0.92 remains. These figures are not confirmed invoices. The 50 Anthropic
cases themselves account for $0.324675 at the configured conservative rates;
some provider refusals may not be billed. No reservation was released.

[Attempt records](attempts.jsonl) contain the 49 newly executed cases, including
full generated response text, outcome/usage metadata, and references to the
already-published frozen inputs. The full response records were published after
explicit user approval and verified against the previously published text hashes
and the local ledger. The first successful Anthropic case remains in
[the workspace-check record](../2026-10-08-anthropic-workspace-check/README.md).
[Verification](verification.json) confirms that SQLite and JSONL agree, source
hashes match, no attempts are in flight, and earlier OpenAI/Gemini records are
unchanged. All 56 focused tests pass. Complete joined exports and the resumable
SQLite ledger are saved locally; credentials are not published.

After this checkpoint, the user approved Gemini credential transfer and the
same worker continued with Gemini. See [the completed Gemini run](../2026-10-08-gemini-completion/README.md) for
the final three-provider status and cleanup. Main generation and Llama-3.1-8B judging have
not started.
See [run metadata](run.json), [infrastructure](infrastructure.json), and the
[cumulative report](report.json).
