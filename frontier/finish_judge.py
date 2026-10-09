"""Join completed output-only judgments to all frozen main cases by request ID.

Run python -m frontier.finish_judge after the three frontier judge jobs finish.
Empty provider blocks retain blank policy labels and an explicit judge_status.
"""
import argparse
import csv
import json
import os
from pathlib import Path

from frontier.prepare import ROOT
from frontier.prepare_judge import FIELDS, _sha
from scoring.llm_policy_run_jobs import load_jobs, validate_job_input
from scoring.llm_policy_runner import LLM_COLS
from scoring.llm_policy_schema import VALID_LABELS


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def finish(prepared_dir=ROOT / ".local/frontier/judge-main", output_path=None):
    prepared_dir = Path(prepared_dir)
    output_path = Path(output_path) if output_path else prepared_dir / "labeled_all_cases.csv"
    summary = json.loads((prepared_dir / "summary.json").read_text())
    source = prepared_dir / "all_outcomes.csv"
    if _sha(source) != summary["all_outcomes_sha256"]:
        raise ValueError("Prepared all_outcomes.csv changed; prepare the judge inputs again")
    all_rows = _read(source)
    by_id = {row["request_id"]: row for row in all_rows}
    if len(by_id) != len(all_rows) or len(all_rows) != summary["expected_cases"]:
        raise ValueError("Prepared cases have duplicate IDs or incorrect coverage")
    judge_models = {summary["judge"], summary["judge"].replace("/", "--")}
    judgments = {}
    for job in load_jobs(prepared_dir / "jobs.yaml"):
        job = {**job, "input": _resolve(job["input"]), "output_dir": _resolve(job["output_dir"])}
        validate_job_input(job)
        output = job["output_dir"]
        manifest = json.loads((output / "manifest.json").read_text())
        if (not manifest.get("complete") or manifest.get("model") not in judge_models
                or manifest.get("job_id") != job["job_id"]
                or manifest.get("input_sha256") != job["input_sha256"]):
            raise ValueError(f"{job['job_id']}: judge incomplete or provenance mismatch; resubmit with --resume")
        expected = {row["request_id"]: row for row in _read(job["input"])}
        labeled = _read(output / "labeled.csv")
        seen = set()
        for row in labeled:
            key = row.get("request_id")
            if key in seen or key not in expected or key in judgments:
                raise ValueError(f"{job['job_id']}: duplicate or foreign judgment: {key}")
            seen.add(key)
            original = expected[key]
            if (any(row.get(field) != original[field] or original[field] != by_id[key][field]
                    for field in FIELDS)
                    or row.get("llm_policy_label") not in VALID_LABELS
                    or row.get("llm_judge_model") != manifest["model"]
                    or row.get("llm_schema_version") != "v1"
                    or row.get("llm_prompt_set_version") != "v1"):
                raise ValueError(f"{job['job_id']}: invalid label or altered input: {key}")
            judgments[key] = {field: row.get(field, "") for field in LLM_COLS}
        if seen != set(expected):
            raise ValueError(f"{job['job_id']}: missing judged requests; resubmit with --resume")
    eligible = {row["request_id"] for row in all_rows if row["model_output"].strip()}
    if set(judgments) != eligible or len(judgments) != summary["judge_rows"]:
        raise ValueError("Judgments do not cover every nonempty main response")
    combined = []
    for row in all_rows:
        key = row["request_id"]
        if key not in judgments and row["transport_outcome"] != "blocked":
            raise ValueError(f"Unexpected empty outcome: {key}")
        combined.append({**row, **judgments.get(key, {}),
                         "judge_status": "judged" if key in judgments else "not_applicable_empty_provider_block"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*FIELDS, *LLM_COLS, "judge_status"])
        writer.writeheader()
        writer.writerows(combined)
    os.replace(temp, output_path)
    return {"output": str(output_path), "total_cases": len(combined),
            "judged": len(judgments), "empty_provider_blocks": len(combined) - len(judgments)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / ".local/frontier/judge-main")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(finish(args.prepared_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
