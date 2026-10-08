"""Summarize an exported pilot without judging model behavior or exposing answers."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from frontier.prepare import ROOT, validate_manifest
from frontier.run import cases, validate_config


def report(records, manifest, config):
    expected = {c["request_id"]: c for c in cases(manifest, config, "pilot")}
    seen, by_cell, outcomes = set(), defaultdict(list), Counter()
    accounted = 0.0
    for record in records:
        key = record["request_id"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or foreign request in pilot export")
        seen.add(key)
        case = expected[key]
        result = record.get("result") or {}
        outcomes[result.get("outcome", "unresolved")] += 1
        accounted += record["accounted_usd"]
        if record["state"] == "done":
            by_cell[(case["model"]["provider"], case["item"]["stratum"])].append(record["accounted_usd"])
    complete = len(seen) == len(expected) and sum(map(len, by_cell.values())) == len(expected)
    projection = {}
    if complete:
        for model in config["models"]:
            provider = model["provider"]
            projection[provider] = round(sum(
                sum(by_cell[(provider, stratum)]) / len(by_cell[(provider, stratum)]) * count * 73
                for stratum, count in manifest["main_quotas"].items()), 4)
    projected_total = round(sum(projection.values()), 4) if projection else None
    return {
        "expected_requests": len(expected), "recorded_requests": len(seen), "complete": complete,
        "accounted_pilot_usd": round(accounted, 6), "outcomes": dict(outcomes),
        "projected_main_usd_by_provider": projection, "projected_main_usd": projected_total,
        "projection_with_50_percent_headroom_usd": round(projected_total * 1.5, 4) if projection else None,
        "main_generation_allocation_usd": config["main_generation_allocation_usd"],
        "projection_note": "Reweighted by main stratum counts. Five pilot conditions may not predict all 73; 50% headroom is a planning heuristic, not a confidence bound.",
        "next_step": "Review cost, truncations, provider blocks, settings and Runpod spend before creating main-run configuration",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / ".local/frontier/pilot.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/frontier/manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/frontier-pilot.json")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    config = validate_config(json.loads(args.config.read_text()))
    print(json.dumps(report(records, manifest, config), indent=2))


if __name__ == "__main__":
    main()
