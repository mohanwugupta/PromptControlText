"""Local fail-safe that deletes only the pilot pod at its recorded deadline.

Run alongside the pilot controller. Normal completion deletes the pod sooner.
The host must remain awake; this is an API-driven local guard, not a claimed
Runpod server-side schedule. It never copies the Runpod control key to the pod.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from frontier.prepare import ROOT
from scripts.credentials import credential_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    pod_id = state["pod_id"]
    deadline = state["deadline_epoch"]
    if not pod_id.isalnum() or not 0 < deadline - time.time() <= 3 * 3600:
        raise SystemExit("Invalid pilot ID or deadline")
    environment = credential_environment(ROOT / ".env")
    headers = {"Authorization": "Bearer " + environment["RUNPOD_API_KEY"],
               "User-Agent": "PromptControlText-frontier/1"}
    print(json.dumps({"watchdog": "armed", "pod_id": pod_id, "deadline_epoch": deadline}), flush=True)
    while time.time() < deadline:
        if json.loads(args.state.read_text()).get("terminated"):
            print('{"watchdog":"normal_cleanup_confirmed"}', flush=True)
            return
        time.sleep(min(30, max(0, deadline - time.time())))
    for attempt in range(10):
        try:
            request = urllib.request.Request("https://rest.runpod.io/v1/pods/" + pod_id,
                                             method="DELETE", headers=headers)
            with urllib.request.urlopen(request, timeout=30):
                pass
            print('{"watchdog":"deadline_deleted_pod"}', flush=True)
            return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                print('{"watchdog":"pod_already_absent"}', flush=True)
                return
            print(json.dumps({"watchdog": "delete_retry", "http_status": error.code}), flush=True)
        except OSError:
            print('{"watchdog":"delete_retry_network_error"}', flush=True)
        time.sleep(15)
    raise SystemExit("Watchdog could not confirm deletion; immediate manual cleanup required")


if __name__ == "__main__":
    main()
