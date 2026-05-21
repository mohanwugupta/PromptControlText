"""
scoring/llm_policy_runner.py

Run one CSV job through the three-pass LLM judge pipeline.

Usage
-----
python -m scoring.llm_policy_runner \\
  --input artifacts/phase1_results_qwen2_7b.csv \\
  --job-id phase1_qwen2_7b \\
  --output-dir artifacts/llm_policy_labels/phase1_qwen2_7b \\
  --model Qwen2.5-72B-Instruct \\
  --base-url http://localhost:8000/v1 \\
  --batch-size 128 \\
  --resume
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pathlib
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set

import pandas as pd

from models.vllm_client import VLLMClient
from scoring.llm_policy_prompts import (
    get_model_output,
    load_default_prompts,
    load_prompt,
)
from scoring.llm_policy_judge import judge_row
from scoring.llm_policy_adjudicate import resolve_first_pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Column names added to labeled.csv ──────────────────────────────────────
LLM_COLS = [
    "llm_policy_label", "llm_secondary_label", "llm_confidence",
    "llm_resolution_method", "llm_num_agree", "llm_disagreement_type",
    "llm_needs_human_audit", "llm_evidence", "llm_reason",
    "llm_judge_model", "llm_adjudicator_model",
    "llm_schema_version", "llm_prompt_set_version", "llm_parse_error",
]

SCHEMA_VERSION = "v1"
PROMPT_SET_VERSION = "v1"


def _row_hash(model_output: str) -> str:
    return hashlib.sha256(model_output.encode("utf-8")).hexdigest()


def _load_completed_hashes(judge_votes_path: pathlib.Path) -> Set[str]:
    """Return row hashes that already have 3 complete first-pass votes."""
    if not judge_votes_path.exists():
        return set()
    try:
        df = pd.read_csv(judge_votes_path)
        counts = df.groupby("row_hash")["judge_id"].count()
        return set(counts[counts >= 3].index.tolist())
    except Exception:
        return set()


def _append_rows(path: pathlib.Path, rows: List[Dict], fieldnames: List[str]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _build_audit_sample(labeled_df: pd.DataFrame, n_per_stratum: int = 50) -> pd.DataFrame:
    """Build a stratified audit sample."""
    parts = []
    for label in labeled_df["llm_policy_label"].dropna().unique():
        subset = labeled_df[labeled_df["llm_policy_label"] == label]
        parts.append(subset.sample(min(n_per_stratum, len(subset)), random_state=42))
    # Extra: disagreement, low-confidence, human-audit flagged
    disagree = labeled_df[labeled_df["llm_disagreement_type"].isin(
        ["soft_2_1", "hard_2_1", "1_1_1", "adjudicated"])]
    parts.append(disagree.sample(min(100, len(disagree)), random_state=42))
    low_conf = labeled_df[labeled_df["llm_confidence"].fillna(1.0) < 0.70]
    parts.append(low_conf.sample(min(100, len(low_conf)), random_state=42))
    human_audit = labeled_df[labeled_df["llm_needs_human_audit"] == True]
    parts.append(human_audit.sample(min(100, len(human_audit)), random_state=42))
    parts.append(labeled_df.sample(min(100, len(labeled_df)), random_state=99))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).drop_duplicates()


def run_job(
    *,
    input_path: str | pathlib.Path,
    job_id: str,
    output_dir: str | pathlib.Path,
    model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_a: Optional[str] = None,
    prompt_b: Optional[str] = None,
    prompt_c: Optional[str] = None,
    adjudicator_prompt: Optional[str] = None,
    batch_size: int = 128,
    temperature: float = 0.0,
    max_tokens: int = 250,
    max_workers: int = 16,
    resume: bool = True,
    limit: Optional[int] = None,
    sample_strategy: Optional[str] = None,
    stratify_by: Optional[str] = None,
) -> None:
    input_path = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── File paths ────────────────────────────────────────────────────────
    labeled_path       = output_dir / "labeled.csv"
    labels_only_path   = output_dir / "labels_only.csv"
    votes_path         = output_dir / "judge_votes.csv"
    adj_path           = output_dir / "adjudication_votes.csv"
    manifest_path      = output_dir / "manifest.json"
    log_path           = output_dir / "run.log"
    parse_errors_path  = output_dir / "parse_errors.csv"
    audit_path         = output_dir / "audit_sample.csv"
    summary_label_path = output_dir / "summary_by_label.csv"
    summary_dis_path   = output_dir / "summary_by_disagreement.csv"

    # Add file handler to logger for this job
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)

    logger.info("Job %s started.  Input: %s", job_id, input_path)
    t0 = time.time()

    # ── Load prompts ──────────────────────────────────────────────────────
    prompts = load_default_prompts()
    if prompt_a:
        prompts["A"] = load_prompt(prompt_a)
    if prompt_b:
        prompts["B"] = load_prompt(prompt_b)
    if prompt_c:
        prompts["C"] = load_prompt(prompt_c)
    if adjudicator_prompt:
        prompts["adjudicator"] = load_prompt(adjudicator_prompt)

    # Save prompt copies
    for key, fname in [("A", "judge_prompt_A.txt"), ("B", "judge_prompt_B.txt"),
                       ("C", "judge_prompt_C.txt"), ("adjudicator", "adjudicator_prompt.txt")]:
        (output_dir / fname).write_text(prompts[key])

    # ── Load input CSV ────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    # Sampling / limit
    if sample_strategy == "stratified" and stratify_by and limit:
        cols = [c.strip() for c in stratify_by.split(",") if c.strip() in df.columns]
        if cols:
            df = (df.groupby(cols, group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), max(1, limit // df[cols].drop_duplicates().shape[0])), random_state=42)))
            df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
    elif limit:
        df = df.head(limit)

    # Resume: skip completed rows
    completed_hashes: Set[str] = set()
    if resume:
        completed_hashes = _load_completed_hashes(votes_path)
        logger.info("Resuming: %d rows already complete.", len(completed_hashes))

    # ── Client ────────────────────────────────────────────────────────────
    client = VLLMClient(
        model_name=model,
        base_url=base_url,
        max_retries=2,
        enable_cache=False,
    )

    # ── Vote column definitions ───────────────────────────────────────────
    vote_fields = [
        "job_id", "row_index", "row_hash", "judge_id", "judge_prompt_variant",
        "judge_model", "primary_label", "secondary_label", "confidence",
        "contains_answer", "contains_refusal", "contains_clarifying_question",
        "contains_safe_redirect", "mentions_instruction_priority",
        "treats_external_text_as_data", "evidence", "reason",
        "raw_json", "parse_error",
    ]
    adj_fields = [
        "job_id", "row_index", "row_hash", "original_vote_labels",
        "adjudicator_1_label", "adjudicator_2_label", "adjudicator_3_label",
        "final_label", "final_confidence", "adjudication_reason", "needs_human_audit",
    ]
    parse_err_fields = ["job_id", "row_index", "row_hash", "judge_id", "parse_error", "model_output_snippet"]

    # ── Per-row processor ─────────────────────────────────────────────────
    def process_row(idx_row):
        idx, row = idx_row
        row_dict = row.to_dict()
        try:
            model_output = get_model_output(row_dict)
        except ValueError as e:
            logger.warning("Row %d: skipped — %s", idx, e)
            return idx, None, None, None

        rh = _row_hash(model_output)
        if rh in completed_hashes:
            return idx, None, None, None  # already done

        votes = judge_row(
            row_index=idx,
            job_id=job_id,
            model_output=model_output,
            prompts=prompts,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        resolution, adj_rows = resolve_first_pass(
            row_index=idx,
            job_id=job_id,
            model_output=model_output,
            votes=votes,
            prompts=prompts,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return idx, votes, resolution, adj_rows

    # ── Main loop ─────────────────────────────────────────────────────────
    rows_to_process = [(idx, row) for idx, row in df.iterrows()
                       if _row_hash(str(row.get("model_output", "") or row.get("model_out", "")))
                       not in completed_hashes]
    logger.info("%d rows to process.", len(rows_to_process))

    all_resolutions: Dict[int, Dict] = {}
    parse_error_rows: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_row, item): item[0] for item in rows_to_process}
        done = 0
        for fut in as_completed(futures):
            idx, votes, resolution, adj_rows = fut.result()
            done += 1
            if votes is None:
                continue

            # Write votes incrementally
            _append_rows(votes_path, votes, vote_fields)
            if adj_rows:
                _append_rows(adj_path, adj_rows, adj_fields)

            # Collect parse errors
            for v in votes:
                if v.get("parse_error"):
                    row_dict = df.iloc[idx].to_dict()
                    snippet = str(row_dict.get("model_output", ""))[:120]
                    parse_error_rows.append({
                        "job_id": job_id, "row_index": idx,
                        "row_hash": v["row_hash"], "judge_id": v["judge_id"],
                        "parse_error": v["parse_error"], "model_output_snippet": snippet,
                    })

            all_resolutions[idx] = resolution

            if done % 500 == 0:
                logger.info("Progress: %d / %d rows processed.", done, len(rows_to_process))

    # ── Build labeled.csv ─────────────────────────────────────────────────
    result_df = df.copy()
    for col in LLM_COLS:
        result_df[col] = None

    for idx, res in all_resolutions.items():
        for col, val in res.items():
            result_df.at[idx, col] = val
        result_df.at[idx, "llm_judge_model"] = model
        result_df.at[idx, "llm_adjudicator_model"] = model
        result_df.at[idx, "llm_schema_version"] = SCHEMA_VERSION
        result_df.at[idx, "llm_prompt_set_version"] = PROMPT_SET_VERSION
        result_df.at[idx, "llm_parse_error"] = any(
            v.get("parse_error") for v in []  # parse errors already captured
        )

    result_df.to_csv(labeled_path, index=False)
    logger.info("Saved labeled.csv (%d rows).", len(result_df))

    # labels_only.csv
    id_cols = [c for c in ["item_id", "row_index"] if c in result_df.columns]
    result_df[id_cols + LLM_COLS].to_csv(labels_only_path, index=False)

    # parse_errors.csv
    _append_rows(parse_errors_path, parse_error_rows, parse_err_fields)

    # summaries
    if "llm_policy_label" in result_df.columns:
        result_df["llm_policy_label"].value_counts().rename_axis("label").reset_index(name="count").to_csv(
            summary_label_path, index=False)
        result_df["llm_disagreement_type"].value_counts().rename_axis("type").reset_index(name="count").to_csv(
            summary_dis_path, index=False)

    # audit sample
    try:
        audit_df = _build_audit_sample(result_df)
        audit_df.to_csv(audit_path, index=False)
        logger.info("Audit sample: %d rows.", len(audit_df))
    except Exception as e:
        logger.warning("Audit sample failed: %s", e)

    # manifest
    elapsed = time.time() - t0
    manifest = {
        "job_id": job_id,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "model": model,
        "schema_version": SCHEMA_VERSION,
        "prompt_set_version": PROMPT_SET_VERSION,
        "total_rows": len(df),
        "processed_rows": len(all_resolutions),
        "parse_errors": len(parse_error_rows),
        "elapsed_seconds": round(elapsed, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Job %s complete in %.0fs.", job_id, elapsed)
    logger.removeHandler(fh)


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run one CSV through the LLM policy judge.")
    p.add_argument("--input", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--prompt-a")
    p.add_argument("--prompt-b")
    p.add_argument("--prompt-c")
    p.add_argument("--adjudicator-prompt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-strategy", default=None)
    p.add_argument("--stratify-by", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_job(
        input_path=args.input,
        job_id=args.job_id,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        prompt_c=args.prompt_c,
        adjudicator_prompt=args.adjudicator_prompt,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        resume=args.resume,
        limit=args.limit,
        sample_strategy=args.sample_strategy,
        stratify_by=args.stratify_by,
    )
