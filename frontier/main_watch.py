"""Continue bounded main-run sessions on the same worker and ledger.

Waits for the initial launch to finish, then resumes only a clean bounded exit.
Stops on completion, budget/error review, or ten minutes before cleanup.
"""
import argparse
import json
import subprocess
import sys
import time

from frontier.prepare import ROOT
from frontier.run import run_lock


def next_action(exit_code, summary, deadline, now):
    if exit_code != 0:
        return "needs_review"
    if summary.get("complete"):
        return "complete"
    if now + 600 >= deadline:
        return "deadline"
    return "resume"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.deadline - time.time() <= 48 * 3600:
        raise SystemExit("Invalid supervisor deadline")
    out = ROOT / ".local/frontier/main-20261008"
    status_path = out / "supervisor-status.json"

    def status(state, **extra):
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"state": state, "updated_epoch": time.time(),
                                   "deadline_epoch": args.deadline, **extra}) + "\n")
        tmp.replace(status_path)

    with run_lock(out / "supervisor.lock"):
        status("waiting_for_initial_session")
        initial_exit = out / "exit-code"
        while not initial_exit.exists():
            if time.time() + 600 >= args.deadline:
                status("deadline")
                return
            time.sleep(30)
        code = int(initial_exit.read_text())
        while True:
            summary = json.loads((out / "status.json").read_text())
            action = next_action(code, summary, args.deadline, time.time())
            if action != "resume":
                status(action, last_exit_code=code)
                return
            minutes = min(120, max(1, int((args.deadline - time.time() - 600) / 60)))
            status("running", session_minutes=minutes)
            code = subprocess.run([sys.executable, "-u", "-m", "frontier.main_run",
                                   "--watch-minutes", str(minutes)], cwd=ROOT).returncode


if __name__ == "__main__":
    main()
