"""
scoring/llm_policy_run_jobs.py

Run many CSV jobs from a YAML job registry.

Usage
-----
python -m scoring.llm_policy_run_jobs \\
  --jobs configs/llm_policy_jobs.yaml \\
  --model Qwen2.5-72B-Instruct \\
  --base-url http://localhost:8000/v1 \\
  --batch-size 128 \\
  --resume
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List

import yaml

from scoring.llm_policy_runner import run_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_jobs(path: str | pathlib.Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config.get("jobs", [])


def combine_labeled_outputs(
    jobs: List[Dict[str, Any]], output_path: str | pathlib.Path
) -> int:
    """Stream all paper-run labeled CSVs into one analysis-ready file."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    sources = []
    combined_fields = []
    for job in jobs:
        job_id = job["job_id"]
        labeled_path = pathlib.Path(job["output_dir"]) / "labeled.csv"
        if not labeled_path.exists():
            raise FileNotFoundError(
                f"Cannot combine {job_id}: {labeled_path} does not exist"
            )
        with labeled_path.open("r", encoding="utf-8", newline="") as input_file:
            source_fields = csv.DictReader(input_file).fieldnames or []
        for field in source_fields:
            if field not in combined_fields:
                combined_fields.append(field)
        sources.append((job, labeled_path))

    rows_written = 0
    with temp_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["experiment_group", "source_run", *combined_fields],
        )
        writer.writeheader()
        for job, labeled_path in sources:
            job_id = job["job_id"]
            with labeled_path.open("r", encoding="utf-8", newline="") as input_file:
                reader = csv.DictReader(input_file)
                experiment_group = job.get(
                    "experiment_group",
                    "control" if job_id.startswith("control_") else "prompted",
                )
                for row in reader:
                    writer.writerow(
                        {
                            "experiment_group": experiment_group,
                            "source_run": job_id,
                            **row,
                        }
                    )
                    rows_written += 1

    os.replace(temp_path, output_path)
    logger.info("Combined %d rows into %s", rows_written, output_path)
    return rows_written


def run_all_jobs(
    jobs_path: str | pathlib.Path,
    *,
    model: str,
    base_url: str = "http://localhost:8000/v1",
    batch_size: int = 128,
    temperature: float = 0.0,
    max_tokens: int = 250,
    max_workers: int = 16,
    resume: bool = True,
    force: bool = False,
    combined_output: str | pathlib.Path | None = None,
) -> None:
    jobs = load_jobs(jobs_path)
    logger.info("Found %d jobs in %s", len(jobs), jobs_path)

    failed: List[str] = []
    for job in jobs:
        job_id = job["job_id"]
        input_path = pathlib.Path(job["input"])
        output_dir = pathlib.Path(job["output_dir"])
        manifest = output_dir / "manifest.json"

        if not force and manifest.exists():
            logger.info("Job %s already complete (manifest found). Skipping.", job_id)
            continue

        if not input_path.exists():
            logger.error("Job %s: input file not found: %s. Skipping.", job_id, input_path)
            failed.append(job_id)
            continue

        logger.info("Starting job %s …", job_id)
        try:
            run_job(
                input_path=input_path,
                job_id=job_id,
                output_dir=output_dir,
                model=model,
                base_url=base_url,
                batch_size=batch_size,
                temperature=temperature,
                max_tokens=max_tokens,
                max_workers=max_workers,
                resume=resume,
            )
        except Exception as e:
            logger.error("Job %s FAILED: %s", job_id, e, exc_info=True)
            failed.append(job_id)

    if failed:
        logger.error("Failed jobs: %s", failed)
        sys.exit(1)
    else:
        logger.info("All jobs complete.")
        if combined_output:
            combine_labeled_outputs(jobs, combined_output)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run many CSV jobs from a YAML registry.")
    p.add_argument("--jobs", required=True, help="Path to llm_policy_jobs.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run even completed jobs.")
    p.add_argument(
        "--combined-output",
        help="After all jobs finish, combine labeled.csv files at this path.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_all_jobs(
        jobs_path=args.jobs,
        model=args.model,
        base_url=args.base_url,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        resume=args.resume,
        force=args.force,
        combined_output=args.combined_output,
    )
