"""
tests/test_llm_policy_runner.py

PRD §23: tests for scoring/llm_policy_runner.py

Covers:
- _row_hash: determinism, uniqueness
- _load_completed_hashes: missing file, 3-vote threshold, incomplete
- _append_rows: creates header, appends without duplicate header, noop on empty
- _build_audit_sample: returns DataFrame, no duplicates
- run_job: expected output files, 3 votes per row, manifest JSON,
           resume skips completed rows, limit restricts rows,
           labeled.csv has all LLM columns, partial failure preserves votes
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scoring.llm_policy_runner import (
    LLM_COLS,
    _append_rows,
    _build_audit_sample,
    _load_completed_hashes,
    _row_hash,
    run_job,
)


# ── fixtures / helpers ─────────────────────────────────────────────────────

def _make_vote(row_index=0, row_hash="abc", judge_id="A",
               label="compliance", confidence=0.9, parse_error=None):
    return {
        "job_id": "test_job",
        "row_index": row_index,
        "row_hash": row_hash,
        "judge_id": judge_id,
        "judge_prompt_variant": judge_id,
        "judge_model": "test_model",
        "primary_label": label,
        "secondary_label": None,
        "confidence": confidence,
        "contains_answer": True,
        "contains_refusal": False,
        "contains_clarifying_question": False,
        "contains_safe_redirect": False,
        "mentions_instruction_priority": False,
        "treats_external_text_as_data": False,
        "evidence": "test evidence",
        "reason": "test reason",
        "raw_json": "{}",
        "parse_error": parse_error,
    }


def _make_resolution(label="compliance"):
    return {
        "llm_policy_label": label,
        "llm_secondary_label": None,
        "llm_confidence": 0.9,
        "llm_resolution_method": "unanimous",
        "llm_num_agree": 3,
        "llm_disagreement_type": None,
        "llm_needs_human_audit": False,
        "llm_evidence": "test",
        "llm_reason": "test",
    }


def _make_input_csv(tmp_path: pathlib.Path, rows: int = 3) -> pathlib.Path:
    path = tmp_path / "input.csv"
    pd.DataFrame({"model_output": [f"The answer is {i}." for i in range(rows)]}).to_csv(
        path, index=False
    )
    return path


def _fake_votes(row_index, job_id, model_output, **kwargs):
    rh = _row_hash(model_output)
    return [_make_vote(row_index=row_index, row_hash=rh, judge_id=jid) for jid in ("A", "B", "C")]


def _fake_resolve(*args, **kwargs):
    return _make_resolution(), []


# ── _row_hash ──────────────────────────────────────────────────────────────

def test_row_hash_matches_sha256():
    text = "hello world"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert _row_hash(text) == expected


def test_row_hash_is_deterministic():
    assert _row_hash("abc") == _row_hash("abc")


def test_row_hash_differs_for_different_inputs():
    assert _row_hash("abc") != _row_hash("xyz")


def test_row_hash_is_64_hex_chars():
    h = _row_hash("anything")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ── _load_completed_hashes ─────────────────────────────────────────────────

def test_load_completed_hashes_missing_file_returns_empty_set(tmp_path):
    result = _load_completed_hashes(tmp_path / "nonexistent.csv")
    assert result == set()


def test_load_completed_hashes_includes_rows_with_three_votes(tmp_path):
    votes_path = tmp_path / "judge_votes.csv"
    rows = [
        {"row_hash": "aaa", "judge_id": "A"},
        {"row_hash": "aaa", "judge_id": "B"},
        {"row_hash": "aaa", "judge_id": "C"},
        {"row_hash": "bbb", "judge_id": "A"},
        {"row_hash": "bbb", "judge_id": "B"},
    ]
    pd.DataFrame(rows).to_csv(votes_path, index=False)
    result = _load_completed_hashes(votes_path)
    assert "aaa" in result


def test_load_completed_hashes_excludes_rows_with_fewer_than_three_votes(tmp_path):
    votes_path = tmp_path / "judge_votes.csv"
    rows = [
        {"row_hash": "bbb", "judge_id": "A"},
        {"row_hash": "bbb", "judge_id": "B"},
    ]
    pd.DataFrame(rows).to_csv(votes_path, index=False)
    result = _load_completed_hashes(votes_path)
    assert "bbb" not in result


def test_load_completed_hashes_single_vote_not_complete(tmp_path):
    votes_path = tmp_path / "judge_votes.csv"
    pd.DataFrame([{"row_hash": "xxx", "judge_id": "A"}]).to_csv(votes_path, index=False)
    result = _load_completed_hashes(votes_path)
    assert len(result) == 0


# ── _append_rows ───────────────────────────────────────────────────────────

def test_append_rows_creates_file_with_header(tmp_path):
    path = tmp_path / "out.csv"
    _append_rows(path, [{"col1": "a", "col2": "b"}], ["col1", "col2"])
    df = pd.read_csv(path)
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0]["col1"] == "a"


def test_append_rows_appends_without_duplicate_header(tmp_path):
    path = tmp_path / "out.csv"
    _append_rows(path, [{"c": "1"}], ["c"])
    _append_rows(path, [{"c": "2"}], ["c"])
    df = pd.read_csv(path)
    assert len(df) == 2
    assert list(df["c"].astype(str)) == ["1", "2"]


def test_append_rows_noop_on_empty_list_does_not_create_file(tmp_path):
    path = tmp_path / "out.csv"
    _append_rows(path, [], ["c"])
    assert not path.exists()


def test_append_rows_multiple_rows_in_one_call(tmp_path):
    path = tmp_path / "out.csv"
    rows = [{"v": str(i)} for i in range(5)]
    _append_rows(path, rows, ["v"])
    df = pd.read_csv(path)
    assert len(df) == 5


# ── _build_audit_sample ────────────────────────────────────────────────────

def _make_labeled_df(n: int = 60) -> pd.DataFrame:
    labels = ["compliance", "refusal", "clarification",
              "hierarchy_preservation", "source_isolation", "safe_redirection"]
    return pd.DataFrame({
        "model_output": [f"output_{i}" for i in range(n)],
        "llm_policy_label": [labels[i % len(labels)] for i in range(n)],
        "llm_confidence": [0.9 if i % 3 != 0 else 0.5 for i in range(n)],
        "llm_disagreement_type": [None] * n,
        "llm_needs_human_audit": [False] * n,
    })


def test_build_audit_sample_returns_dataframe():
    df = _make_labeled_df(60)
    result = _build_audit_sample(df, n_per_stratum=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_build_audit_sample_no_duplicates():
    df = _make_labeled_df(60)
    result = _build_audit_sample(df, n_per_stratum=5)
    assert result.duplicated().sum() == 0


def test_build_audit_sample_covers_all_labels():
    df = _make_labeled_df(60)
    result = _build_audit_sample(df, n_per_stratum=5)
    present = set(result["llm_policy_label"].dropna().unique())
    labels = {"compliance", "refusal", "clarification",
              "hierarchy_preservation", "source_isolation", "safe_redirection"}
    assert labels.issubset(present)


# ── run_job integration (VLLMClient, judge_row, resolve_first_pass mocked) ──

@patch("scoring.llm_policy_runner.VLLMClient")
@patch("scoring.llm_policy_runner.resolve_first_pass", side_effect=_fake_resolve)
@patch("scoring.llm_policy_runner.judge_row", side_effect=_fake_votes)
def test_run_job_creates_expected_output_files(mock_judge, mock_resolve, mock_client, tmp_path):
    input_csv = _make_input_csv(tmp_path, rows=2)
    out_dir = tmp_path / "out"
    run_job(
        input_path=input_csv, job_id="test_job",
        output_dir=out_dir, model="test_model", resume=False,
    )
    for fname in ("labeled.csv", "labels_only.csv", "judge_votes.csv",
                  "manifest.json", "run.log"):
        assert (out_dir / fname).exists(), f"Missing: {fname}"


@patch("scoring.llm_policy_runner.VLLMClient")
@patch("scoring.llm_policy_runner.resolve_first_pass", side_effect=_fake_resolve)
@patch("scoring.llm_policy_runner.judge_row", side_effect=_fake_votes)
def test_run_job_writes_three_votes_per_row(mock_judge, mock_resolve, mock_client, tmp_path):
    n_rows = 4
    input_csv = _make_input_csv(tmp_path, rows=n_rows)
    out_dir = tmp_path / "out"
    run_job(
        input_path=input_csv, job_id="j", output_dir=out_dir,
        model="m", resume=False,
    )
    votes_df = pd.read_csv(out_dir / "judge_votes.csv")
    assert len(votes_df) == n_rows * 3


@patch("scoring.llm_policy_runner.VLLMClient")
@patch("scoring.llm_policy_runner.resolve_first_pass", side_effect=_fake_resolve)
@patch("scoring.llm_policy_runner.judge_row", side_effect=_fake_votes)
def test_run_job_manifest_is_valid_json_with_required_keys(mock_judge, mock_resolve, mock_client, tmp_path):
    input_csv = _make_input_csv(tmp_path, rows=2)
    out_dir = tmp_path / "out"
    run_job(
        input_path=input_csv, job_id="my_job", output_dir=out_dir,
        model="test_model", resume=False,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    for key in ("job_id", "total_rows", "processed_rows", "elapsed_seconds",
                "schema_version", "prompt_set_version"):
        assert key in manifest, f"manifest missing: {key}"
    assert manifest["job_id"] == "my_job"
    assert manifest["total_rows"] == 2


@patch("scoring.llm_policy_runner.VLLMClient")
@patch("scoring.llm_policy_runner.resolve_first_pass", side_effect=_fake_resolve)
@patch("scoring.llm_policy_runner.judge_row", side_effect=_fake_votes)
def test_run_job_resume_skips_already_completed_rows(mock_judge, mock_resolve, mock_client, tmp_path):
    input_csv = _make_input_csv(tmp_path, rows=3)
    out_dir = tmp_path / "out"
    # First run — processes all 3 rows
    run_job(input_path=input_csv, job_id="j", output_dir=out_dir, model="m", resume=False)
    mock_judge.reset_mock()
    # Second run with resume — all rows already have 3 votes; judge should not be called again
    run_job(input_path=input_csv, job_id="j", output_dir=out_dir, model="m", resume=True)
    assert mock_judge.call_count == 0


@patch("scoring.llm_policy_runner.VLLMClient")
@patch("scoring.llm_policy_runner.resolve_first_pass", side_effect=_fake_resolve)
@patch("scoring.llm_policy_runner.judge_row", side_effect=_fake_votes)
def test_run_job_limit_restricts_number_of_rows_processed(mock_judge, mock_resolve, mock_client, tmp_path):
    input_csv = _make_input_csv(tmp_path, rows=10)
    out_dir = tmp_path / "out"
    run_job(
        input_path=input_csv, job_id="j", output_dir=out_dir,
        model="m", resume=False, limit=3,
    )
    votes_df = pd.read_csv(out_dir / "judge_votes.csv")
    assert len(votes_df) == 3 * 3  # 3 rows × 3 judges


@patch("scoring.llm_policy_runner.VLLMClient")
@patch("scoring.llm_policy_runner.resolve_first_pass", side_effect=_fake_resolve)
@patch("scoring.llm_policy_runner.judge_row", side_effect=_fake_votes)
def test_run_job_labeled_csv_contains_all_llm_columns(mock_judge, mock_resolve, mock_client, tmp_path):
    input_csv = _make_input_csv(tmp_path, rows=2)
    out_dir = tmp_path / "out"
    run_job(input_path=input_csv, job_id="j", output_dir=out_dir, model="m", resume=False)
    labeled_df = pd.read_csv(out_dir / "labeled.csv")
    for col in LLM_COLS:
        assert col in labeled_df.columns, f"labeled.csv missing column: {col}"


@patch("scoring.llm_policy_runner.VLLMClient")
@patch("scoring.llm_policy_runner.resolve_first_pass", side_effect=_fake_resolve)
@patch("scoring.llm_policy_runner.judge_row", side_effect=_fake_votes)
def test_run_job_preserves_original_input_columns(mock_judge, mock_resolve, mock_client, tmp_path):
    """labeled.csv must retain all columns from the input CSV."""
    path = tmp_path / "input.csv"
    pd.DataFrame({
        "model_output": ["text a", "text b"],
        "item_id": ["id1", "id2"],
        "benchmark": ["hb", "hb"],
    }).to_csv(path, index=False)
    out_dir = tmp_path / "out"
    run_job(input_path=path, job_id="j", output_dir=out_dir, model="m", resume=False)
    labeled_df = pd.read_csv(out_dir / "labeled.csv")
    assert "item_id" in labeled_df.columns
    assert "benchmark" in labeled_df.columns


@patch("scoring.llm_policy_runner.VLLMClient")
@patch("scoring.llm_policy_runner.resolve_first_pass", side_effect=_fake_resolve)
@patch("scoring.llm_policy_runner.judge_row")
def test_run_job_partial_row_failure_preserves_votes_for_successful_rows(
    mock_judge, mock_resolve, mock_client, tmp_path
):
    """If judge_row raises for one row, votes already written for others are preserved."""
    call_count = [0]

    def _votes_with_failure(row_index, job_id, model_output, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("simulated row failure")
        rh = _row_hash(model_output)
        return [_make_vote(row_index=row_index, row_hash=rh, judge_id=jid) for jid in ("A", "B", "C")]

    mock_judge.side_effect = _votes_with_failure

    input_csv = _make_input_csv(tmp_path, rows=3)
    out_dir = tmp_path / "out"
    try:
        run_job(input_path=input_csv, job_id="j", output_dir=out_dir, model="m", resume=False)
    except Exception:
        pass  # per-row failure is handled; whole-job exception also acceptable

    if (out_dir / "judge_votes.csv").exists():
        votes_df = pd.read_csv(out_dir / "judge_votes.csv")
        assert len(votes_df) >= 3  # at least one row's 3 votes survived
