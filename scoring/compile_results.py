"""Compile existing judgments for analysis, offline, without rerunning any model.

python -m scoring.compile_results
Rebuilds artifacts/phase1_results_combined_labeled.csv from the original eight
jobs, six new open-model jobs, and all verified frontier main cases.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

from frontier.prepare import ROOT
from frontier.prepare_judge import prepare
from frontier.finish_judge import finish
from scoring.llm_policy_run_jobs import load_jobs
from scoring.llm_policy_schema import VALID_LABELS

logger = logging.getLogger(__name__)
EXTRA_FIELDS = ["experiment_group", "source_run", "analysis_study", "model_cohort",
                "judge_status", "score_source", "model_name"]
SCORE_FIELDS = ["malformed_output", "refusal_score", "compliance_score", "clarification_score",
                "abstention_score", "hierarchy_following_score", "unsafe_continuation_score",
                "stop_compliance_score"]
KEY_FIELDS = ["benchmark", "item_id", "prompt_family", "clarity_level", "prompt_variant"]


def _path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def discover_sources(include_frontier=True, prepared_dir=ROOT / ".local/frontier/judge-main"):
    sources = []
    for registry, cohort in [("configs/llm_policy_jobs.yaml", "original_open_models"),
                             ("configs/llm_policy_jobs_new_models.yaml", "new_open_models")]:
        for job in load_jobs(ROOT / registry):
            directory = _path(job["output_dir"])
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest.get("job_id") != job["job_id"]:
                raise ValueError(f"{job['job_id']}: saved judge job ID differs")
            sources.append({"job_id": job["job_id"], "path": directory / "labeled.csv",
                            "expected_rows": job.get("expected_rows", manifest["total_rows"]),
                            "analysis_study": "open_models", "model_cohort": cohort,
                            "experiment_group": job["experiment_group"],
                            "default_model": "Qwen--Qwen2.5-72B-Instruct" if job["job_id"] == "phase1_qwen25_72b" else "",
                            "judge_manifest": manifest})
    if include_frontier:
        prepared_dir = Path(prepared_dir)
        # Regenerate portable, hash-verified inputs from the frozen upload; this
        # also supports fresh checkouts without the cluster's Git-ignored files.
        prepare(output_dir=prepared_dir)
        result = finish(prepared_dir)
        sources.append({"job_id": "frontier_main", "path": Path(result["output"]),
                        "expected_rows": result["total_cases"], "analysis_study": "frontier_main",
                        "model_cohort": "frontier_models"})
    return sources


def _frontier_scores(row):
    # Use exactly the original generation pipeline's output-quality check and
    # heuristic scorers. These are separate from the LLM labels and native task
    # correctness, and require no model or provider call.
    from experiments.run_phase1 import _is_valid_output, score_item
    valid, reason = _is_valid_output(row["model_output"])
    row["malformed_output"] = "" if valid else reason
    if valid:
        row.update({f"{key}_score": value for key, value in score_item(row["benchmark"], row["model_output"]).items()})


def compile_sources(sources, output_path):
    """Stream the union of source columns; preserve missing/error labels explicitly."""
    output_path = Path(output_path)
    paths = [Path(source["path"]).resolve() for source in sources]
    if len(paths) != len(set(paths)) or output_path.resolve() in paths:
        raise ValueError("Sources must be distinct and cannot include the combined output")
    fields = list(EXTRA_FIELDS)
    required = {*KEY_FIELDS, "model_output", "llm_policy_label", "input_text"}
    for source, path in zip(sources, paths):
        with path.open(newline="", encoding="utf-8") as stream:
            header = csv.DictReader(stream).fieldnames or []
        if required - set(header):
            raise ValueError(f"{source['job_id']}: missing required columns {sorted(required - set(header))}")
        fields.extend(field for field in header if field not in fields)
    fields.extend(field for field in SCORE_FIELDS if field not in fields)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(16 * 1024 * 1024)
    report = {"output": str(output_path), "total_rows": 0, "by_source": {},
              "by_study": Counter(), "by_judge_status": Counter()}
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=output_path.parent,
                                         prefix=output_path.name + ".", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for source, path in zip(sources, paths):
                logger.info("Compiling %s ...", source["job_id"])
                before = path.stat()
                seen, count, statuses, labels = set(), 0, Counter(), Counter()
                hasher = hashlib.sha256()
                def lines(raw):
                    for line in raw:
                        hasher.update(line)
                        yield line.decode("utf-8")
                with path.open("rb") as raw:
                    for row in csv.DictReader(lines(raw)):
                        if None in row:
                            raise ValueError(f"{source['job_id']}: malformed CSV record")
                        frontier = source["analysis_study"] == "frontier_main"
                        key = row.get("request_id") if frontier else tuple(row[field] for field in KEY_FIELDS)
                        if not key or key in seen:
                            raise ValueError(f"{source['job_id']}: duplicate or empty case identity: {key}")
                        seen.add(key)
                        label = row["llm_policy_label"]
                        if label in VALID_LABELS:
                            status = "judged"
                        elif frontier and not label and not row["model_output"].strip() and row["transport_outcome"] == "blocked":
                            status = "not_applicable_empty_provider_block"
                        else:
                            status = "missing_judgment" if not label else "invalid_judgment"
                        if frontier and status not in {"judged", "not_applicable_empty_provider_block"}:
                            raise ValueError(f"Unresolved frontier judgment: {key}")
                        row.update(analysis_study=source["analysis_study"], model_cohort=source["model_cohort"],
                                   judge_status=status, source_run=f"frontier_main_{row['provider']}" if frontier else source["job_id"])
                        row["experiment_group"] = row.get("experiment_group") or source.get("experiment_group", "")
                        row["model_name"] = row.get("model_name") or source.get("default_model", "")
                        if not row["model_name"]:
                            raise ValueError(f"{source['job_id']}: missing generator model")
                        if frontier:
                            if status == "judged":
                                _frontier_scores(row)
                                row["score_source"] = "original_heuristic_scorers"
                            else:
                                row["score_source"] = "not_applicable_empty_provider_block"
                        else:
                            row["score_source"] = "original_generation"
                        writer.writerow(row)
                        count += 1
                        statuses[status] += 1
                        labels[label or "missing"] += 1
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Source changed while compiling: {path}")
                if count != source["expected_rows"]:
                    raise ValueError(f"{source['job_id']}: found {count}/{source['expected_rows']} rows")
                report["by_source"][source["job_id"]] = {
                    "rows": count, "judge_status": dict(statuses), "policy_labels": dict(labels),
                    "path": str(path), "sha256": hasher.hexdigest(),
                }
                report["total_rows"] += count
                report["by_study"][source["analysis_study"]] += count
                report["by_judge_status"].update(statuses)
                logger.info("%s: %d rows; %s", source["job_id"], count, dict(statuses))
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
    for key in ("by_study", "by_judge_status"):
        report[key] = dict(report[key])
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/phase1_results_combined_labeled.csv")
    parser.add_argument("--open-models-only", action="store_true", help="Exclude the frontier main experiment")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = compile_sources(discover_sources(include_frontier=not args.open_models_only), args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "by_source"}, indent=2))


if __name__ == "__main__":
    main()
