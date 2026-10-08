# Main frontier run — launch record

**Main generation has started.** All three providers accepted their initial
five-case Batch jobs on October 8 UTC (October 7 Pacific). The target remains
50 fixed items × 73 conditions × three generators = **10,950 cases**.
The [latest published progress](progress.json) records batch IDs, completion
counts, budget accounting and checkpoint verification. The [launch snapshot](launch.json)
preserves the earlier starting state. Both are timestamped snapshots, not live status.

Fifteen complete generated records (five per provider) are verified locally.
Their public release is pending explicit main-study approval after automatic
approval review rejected it; the earlier raw-data approval covered the pilot.
Generation continues while that publication request is pending.
[requests.jsonl](requests.jsonl) accounts for all
615 cases submitted or reserved at this snapshot, including 600 still pending.
Each finished response is linked to its frozen item, condition and request hash.
[operations/](operations/) archives the actual deployment and backup scripts;
[execution-provenance.json](execution-provenance.json) records their hashes.
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

Batch jobs may take up to 24 hours per wave. A supervisor resumes clean two-hour
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
