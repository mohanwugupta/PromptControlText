# Resume after terminal-block accounting review

The remaining Anthropic batch has been retrieved. All 1,615 submitted cases
are now accounted for, including the four terminal Gemini prompt blocks;
9,335 frozen cases remain unsubmitted. No blocked or completed case is resent.
The ledger total remains $6.749424 after the state-only correction. All four
unknown-usage reservations remain held; no usage fields or responses changed.
See [resume-resolution.json](resume-resolution.json) for IDs and checkpoint ancestry.

The updated runner distinguishes terminal prompt blocks from unknown provider
errors. A missing charge does not make an explicit prompt block unfinished:
its full reservation remains included in the $160 cap while other cases proceed.
Missing usage on text answers, unknown block reasons, unexpected input usage,
server errors, and ambiguous submissions still stop new submissions.

Validation: 85 focused tests passed, followed by a 29-test main-run check after
narrowing reconciliation to affected batches. The copied ledger was checked
against all frozen request payloads. Only the four completion states and their
batch state changed; costs, request IDs and all normalized results are identical.
The original checkpoint and published full-record files remain immutable.
The metadata index supplies current states after reconciliation.

Execution will restore this existing ledger onto one replacement CPU worker.
The original October 9 cleanup deadline and $5 total main infrastructure reserve
remain fixed, including the first worker's cost. New submissions remain bounded
at 200 cases per provider, one active batch per provider, within the same $160
cap and $250 study ceiling. The sample, model settings and 8B judge are unchanged.
See [progress.json](progress.json) for the latest verified execution phase;
this preparation record alone does not confirm a running worker.
