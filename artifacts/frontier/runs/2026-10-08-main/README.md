# Main frontier run — deployment record

The main study is **prepared; inference has not started at this checkpoint**.
Its target is 50 fixed items × 73 conditions × three generators = **10,950 cases**.
The [execution plan](../../../../docs/frontier-main-run.md) specifies the design,
Batch transport, accounting assumptions, recovery behavior and caps.

Published code commit: `8db7bdd680c36b5f434b7cc6b5f564c96b22c88a`.
All 15 deployed source files match their recorded hashes. The Runpod dry run
verified 10,950 unique cases and 3,650 per provider; 76 local focused tests passed.

Worker `43dht91rp868ts` uses two CPU cores, 4 GB RAM and the official Ubuntu 24.04
image at $0.06/hour, within a $5 main infrastructure reserve. A local three-hour
API deletion watchdog is armed; it requires the host to remain awake and online.
Unrelated Runpod account resources were left untouched.

Automatic approval review rejected transfer of the three provider API keys and
Anthropic workspace ID to this new worker because prior approvals named a pilot
worker. A bundled approval request is pending. The prepared worker currently has
no provider credentials and has submitted no main requests. Its exact timestamps,
configuration hashes and limits are in [preparation.json](preparation.json).

This record will be supplemented with actual batch IDs, coverage, cost accounting,
retrieval verification and cleanup evidence after execution. It is not a result
report, an invoice, or confirmation that the main study is running.
