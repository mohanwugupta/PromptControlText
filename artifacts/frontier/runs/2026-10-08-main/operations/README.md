# Archived Runpod operation scripts

These are the exact local operation scripts used for this study, archived for
review. They contain credential variable names and file locations, **no values**.
The reusable experiment engine is in `frontier/`; these scripts record deployment,
SSH transfer, launch, checkpoint verification, supervision and cleanup.

They expect execution from the repository root and the existing private
`.local/frontier/main-20261008/` state. They are historical operation code, not a
request to recreate or delete infrastructure. `control.py` reads credentials from
the ignored local `.env` and transfers only the explicitly approved provider keys
to the named worker. The Runpod control key stays local. Private state, SSH keys,
account/workspace identifiers and environment files are excluded from Git.

`monitor.py` backs up every three minutes and deletes only this study's worker
after a verified checkpoint and completion/error/deadline. `extend_runtime.py`
records the guarded transition to the existing $5 reserve's 48-hour limit.
