# Main frontier run — launch record

**Main generation is running.** Snapshot: 2026-10-08T07:44:43.083345+00:00.
The [latest progress](progress.json) and [current metadata](requests.jsonl) report
the verified state at this checkpoint, not a live dashboard. The
[resume launch](resume-launch.json) and [stop report](STOP-REPORT.md) preserve
earlier events. The target remains **10,950 frozen cases**. Judging has not started.

Full generated records available at this checkpoint are in [responses/](responses/):
**9,505 records**. The user explicitly approved public release of all main-study
records as they finish, including the synthetic test codes from the public IHEval
benchmark. [provider-results/](provider-results/) preserves 9,505 original
provider result rows verified against normalized records in this export.
[requests.jsonl](requests.jsonl) accounts for all **9,705 cases** submitted or
reserved at this snapshot: 9,505 completed,
0 needing review, and 200 pending.
Another 1,245 frozen cases have not been submitted.
Each finished response maps to its frozen item, condition and request hash.
[operations/](operations/) archives the deployment, backup and export scripts;
[execution-provenance.json](execution-provenance.json) records execution hashes.
Credentials, SSH keys, private account/workspace configuration, temporary bundles
and duplicate local database backups remain excluded from the public repo.

The user explicitly approved transferring the three provider keys and Anthropic
workspace ID to worker `43dht91rp868ts`. Transfer succeeded over encrypted SSH,
with mode 600 on the credential file. No keys or workspace IDs are published.
The earlier [preparation checkpoint](preparation.json) preserves the original
approval blocker; that blocker is now resolved.

Generator code commit: `8db7bdd680c36b5f434b7cc6b5f564c96b22c88a`.
All 15 originally deployed files matched their hashes. The Runpod dry run verified
10,950 unique cases and 3,650 per provider. The continuation/cleanup update passed
78 focused tests and changes scheduling only; model settings, prompts, request
identities, sample and API cap remain fixed.

The stopped deployment was configured as follows. Batch jobs may take up to
24 hours per wave. A supervisor resumes clean two-hour
polling sessions using the same ledger. It stops on completion, review/budget
errors, or the deadline; ambiguous submissions are never blindly retried.
The worker lifetime is limited to 48 hours from creation, costing at most $2.88
in compute at $0.06/hour within the $5 infrastructure reserve. The local API
watchdog enforces deletion and requires the host to stay awake and connected.
Local checkpoints are retrieved regularly; unrelated Runpod resources are untouched.

The [execution plan](../../../../docs/frontier-main-run.md) documents the $160
API cap within the $250 study ceiling, accounting assumptions and recovery.
A cap or time limit can leave the study incomplete. Already-submitted provider
Batch jobs can continue after worker deletion; their IDs and reserved costs
remain in the saved ledger. Judge validation, judging and analysis are not yet done.
