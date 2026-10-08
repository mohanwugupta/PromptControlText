"""Explicitly reopen one known Anthropic account-setup rejection after an update."""
import argparse
import json
from pathlib import Path

from frontier.prepare import ROOT, validate_manifest
from frontier.run import Ledger, run_lock, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", required=True)
    update = parser.add_mutually_exclusive_group(required=True)
    update.add_argument("--funding-update")
    update.add_argument("--workspace-update")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".local/frontier/pilot.sqlite")
    parser.add_argument("--export", type=Path, default=ROOT / ".local/frontier/pilot.jsonl")
    args = parser.parse_args()
    manifest = validate_manifest(json.loads((ROOT / "artifacts/frontier/manifest.json").read_text()))
    config = validate_config(json.loads((ROOT / "configs/frontier-pilot.json").read_text()))
    with run_lock(args.ledger.with_suffix(".lock")):
        ledger = Ledger(args.ledger, manifest, config)
        if args.workspace_update is not None:
            record = ledger.retry_workspace_anthropic(args.request_id, args.workspace_update)
        else:
            record = ledger.retry_funded_anthropic(args.request_id, args.funding_update)
        ledger.export(args.export)
        print(json.dumps({"status": "retry_ready", "archived_attempt": record,
                          "accounted_usd": round(ledger.total(), 6)}))


if __name__ == "__main__":
    main()
