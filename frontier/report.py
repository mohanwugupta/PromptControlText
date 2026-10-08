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
    unknown_cost_requests, unknown_cost_reservations = 0, 0.0
    unknown_by_provider = Counter()
    prior_attempt_count, prior_attempt_cost = 0, 0.0
    prior_unknown_count, prior_unknown_cost = 0, 0.0
    for record in records:
        key = record["request_id"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or foreign request in pilot export")
        seen.add(key)
        case = expected[key]
        result = record.get("result") or {}
        outcomes[result.get("outcome", "unresolved")] += 1
        accounted += record["accounted_usd"]
        for attempt in record.get("prior_attempts", []):
            prior_attempt_count += 1
            prior_attempt_cost += attempt["accounted_usd"]
            accounted += attempt["accounted_usd"]
            if (attempt.get("result") or {}).get("cost_basis") == "reserved_unknown":
                prior_unknown_count += 1
                prior_unknown_cost += attempt["accounted_usd"]
        if record["state"] == "reserved" or result.get("cost_basis") == "reserved_unknown":
            unknown_cost_requests += 1
            unknown_cost_reservations += record["accounted_usd"]
            unknown_by_provider[case["model"]["provider"]] += 1
        if record["state"] == "done":
            by_cell[(case["model"]["provider"], case["item"]["stratum"])].append(record["accounted_usd"])
    complete = len(seen) == len(expected) and sum(map(len, by_cell.values())) == len(expected)
    projection = {}
    completed_by_provider = {model["provider"]: sum(len(values) for (provider, _), values in by_cell.items()
                                                   if provider == model["provider"])
                             for model in config["models"]}
    for model in config["models"]:
        provider = model["provider"]
        if completed_by_provider[provider] == 50:
            projection[provider] = round(sum(
                sum(by_cell[(provider, stratum)]) / len(by_cell[(provider, stratum)]) * count * 73
                for stratum, count in manifest["main_quotas"].items()), 4)
    accounting_projection = projection
    projection = {provider: value for provider, value in accounting_projection.items()
                  if unknown_by_provider[provider] == 0}
    projected_total = round(sum(projection.values()), 4) if complete and len(projection) == 3 else None
    return {
        "expected_requests": len(expected), "recorded_requests": len(seen), "complete": complete,
        "accounted_pilot_usd": round(accounted, 6), "outcomes": dict(outcomes),
        "prior_attempts": prior_attempt_count, "prior_attempt_accounted_usd": round(prior_attempt_cost, 6),
        "prior_unknown_cost_attempts": prior_unknown_count,
        "prior_unknown_cost_reservations_usd": round(prior_unknown_cost, 6),
        "unknown_cost_requests": unknown_cost_requests,
        "unknown_cost_reservations_usd": round(unknown_cost_reservations, 6),
        "completed_by_provider": completed_by_provider,
        "projected_main_accounting_usd_by_provider": accounting_projection,
        "projected_main_usd_by_provider": projection, "projected_main_usd": projected_total,
        "projection_with_50_percent_headroom_usd": round(projected_total * 1.5, 4) if projected_total is not None else None,
        "main_generation_allocation_usd": config["main_generation_allocation_usd"],
        "projection_note": "Reweighted by main stratum counts. Includes full-token reservations wherever usage is missing; reconcile them before treating this as expected spend. Five pilot conditions may not predict all 73; 50% headroom is a planning heuristic, not a confidence bound.",
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
