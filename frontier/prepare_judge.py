"""Verify the completed main upload and prepare CSVs for the existing LLM judge.

Offline only: no provider calls, generation, or judging. Run from the repository
with python -m frontier.prepare_judge before submitting the frontier judge array.
"""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml

from frontier.main_run import main_cases, reserved_terminal_block, validate_main
from frontier.prepare import ROOT, digest

FIELDS = [
    "request_id", "study_stage", "provider", "model_name", "returned_model",
    "item_id", "group_id", "benchmark", "stratum", "domain", "gold_label",
    "condition_id", "prompt_family", "clarity_level", "prompt_variant",
    "experiment_group", "input_text", "base_system", "reference_json",
    "model_output", "transport_outcome", "provider_refusal", "truncated",
    "finish_reason", "generation_state", "archived_generation_state",
    "request_body_sha256", "response_text_sha256", "source_file",
]


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _portable(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def prepare(run_dir=ROOT / "artifacts/frontier/runs/2026-10-08-main",
            output_dir=ROOT / ".local/frontier/judge-main",
            manifest_path=ROOT / "artifacts/frontier/manifest.json",
            config_path=ROOT / "configs/frontier-main.json",
            pilot_path=ROOT / "configs/frontier-pilot.json"):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    config = validate_main(json.loads(Path(config_path).read_text()), manifest,
                           json.loads(Path(pilot_path).read_text()))
    expected = {case["request_id"]: case for case in main_cases(manifest, config)}
    metadata_rows = _jsonl(run_dir / "requests.jsonl")
    metadata = {row["request_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows) or set(metadata) != set(expected):
        raise ValueError("Request metadata must cover every frozen main case exactly once")
    if any(row["state"] != "done" for row in metadata_rows):
        raise ValueError("Main generation still has pending or review cases")
    progress = json.loads((run_dir / "progress.json").read_text())
    resolved = set(progress.get("terminal_block_resolution", {}).get("request_ids", []))
    all_rows, seen, sources = [], set(), {}
    for path in sorted((run_dir / "responses").glob("*.jsonl")):
        sources[_portable(path)] = _sha(path)
        for record in _jsonl(path):
            key = record["request_id"]
            if key in seen or key not in expected:
                raise ValueError(f"Duplicate or foreign main response: {key}")
            seen.add(key)
            case, result, meta = record["case"], record.get("result") or {}, metadata[key]
            if case != expected[key] or record["provider"] != case["model"]["provider"]:
                raise ValueError(f"Response does not match the frozen case: {key}")
            if record["state"] != "done" and not (
                record["state"] == "needs_review" and key in resolved
                and reserved_terminal_block(case, result)
            ):
                raise ValueError(f"Unresolved response state: {key}")
            text = result.get("text", "")
            if not isinstance(text, str):
                raise ValueError(f"Response text is not a string: {key}")
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if (text_hash != meta.get("response_text_sha256")
                    or case["request_body_sha256"] != meta.get("request_body_sha256")
                    or meta["provider"] != record["provider"]
                    or meta["item_id"] != case["item"]["item_id"]
                    or meta["condition_id"] != case["condition"]["condition_id"]
                    or meta["outcome"] != result.get("outcome")):
                raise ValueError(f"Response metadata/hash mismatch: {key}")
            outcome = result.get("outcome")
            if outcome not in {"text", "blocked", "truncated"} or (not text.strip() and outcome != "blocked"):
                raise ValueError(f"Unexpected generation outcome: {key}")
            item, condition = case["item"], case["condition"]
            all_rows.append({
                "request_id": key, "study_stage": "main", "provider": record["provider"],
                "model_name": case["model"]["model"], "returned_model": result.get("returned_model"),
                "item_id": item["item_id"], "group_id": item["group_id"],
                "benchmark": {"xstest": "XSTest", "harmbench": "HarmBench", "iheval": "IHEval"}[item["benchmark"]],
                "stratum": item["stratum"], "domain": item["domain"], "gold_label": item["gold_label"],
                "condition_id": condition["condition_id"], "prompt_family": condition["family"],
                "clarity_level": condition["clarity"], "prompt_variant": condition["variant"],
                "experiment_group": "control" if condition["text"] is None else "prompted",
                "input_text": item["user_text"], "base_system": item["base_system"],
                "reference_json": json.dumps(item["reference"], ensure_ascii=False),
                "model_output": text, "transport_outcome": outcome,
                "provider_refusal": result.get("provider_refusal", False),
                "truncated": result.get("truncated", False), "finish_reason": result.get("finish_reason"),
                "generation_state": meta["state"], "archived_generation_state": record["state"],
                "request_body_sha256": case["request_body_sha256"], "response_text_sha256": text_hash,
                "source_file": _portable(path),
            })
    if seen != set(expected):
        raise ValueError(f"Incomplete response upload: {len(seen)}/{len(expected)} cases")

    # Validate everything before replacing any previous prepared inputs.
    all_rows.sort(key=lambda row: row["request_id"])
    eligible = defaultdict(list)
    excluded = []
    for row in all_rows:
        if row["model_output"].strip():
            eligible[row["provider"]].append(row)
        else:
            excluded.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "all_outcomes.csv", all_rows)
    _write_csv(output_dir / "excluded_empty_blocks.csv", excluded)
    jobs, by_provider = [], {}
    for model in config["models"]:
        provider = model["provider"]
        rows = eligible[provider]
        path = output_dir / f"{provider}.csv"
        _write_csv(path, rows)
        by_provider[provider] = {
            "model": model["model"], "total_cases": sum(row["provider"] == provider for row in all_rows),
            "judge_rows": len(rows), "empty_blocks": sum(row["provider"] == provider for row in excluded),
            "truncated": sum(bool(row["truncated"]) for row in rows),
        }
        if rows:
            jobs.append({"job_id": f"frontier_main_{provider}", "input": _portable(path),
                         "output_dir": f"artifacts/llm_policy_labels/frontier_main_{provider}",
                         "expected_rows": len(rows), "input_sha256": _sha(path),
                         "require_request_id": True})
    summary = {
        "expected_cases": len(expected), "verified_cases": len(all_rows),
        "judge_rows": sum(len(rows) for rows in eligible.values()), "excluded_empty_blocks": len(excluded),
        "by_provider": by_provider, "manifest_sha256": manifest["manifest_sha256"],
        "config_sha256": digest(config), "source_sha256": sources,
        "metadata_sha256": _sha(run_dir / "requests.jsonl"),
        "all_outcomes_sha256": _sha(output_dir / "all_outcomes.csv"),
        "judge": config["canonical_judge"], "rubric": "existing output-only A/B/C plus adjudication",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "jobs.yaml").write_text(yaml.safe_dump({"jobs": jobs}, sort_keys=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/frontier/runs/2026-10-08-main")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/frontier/judge-main")
    args = parser.parse_args()
    summary = prepare(args.run_dir, args.output_dir)
    print(json.dumps({key: value for key, value in summary.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
