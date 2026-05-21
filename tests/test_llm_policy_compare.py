"""
tests/test_llm_policy_compare.py

PRD §23 / §20: tests for analysis/llm_policy_compare.py

Covers:
- compare() creates disagreement matrix CSV and heatmap PNG
- disagreement matrix has correct 6×6 shape
- bar charts are produced for prompt_family, clarity_level, benchmark
- family_clarity_llm_policy.csv is created with correct columns
- missing rule or LLM column is handled gracefully (no crash, file not created)
- output directory is created if it doesn't exist
- family×clarity counts are all positive
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from analysis.llm_policy_compare import LLM_COL, RULE_COL, VALID_LABELS, compare


# ── fixture helper ─────────────────────────────────────────────────────────

def _make_combined_csv(tmp_path: pathlib.Path, n: int = 30) -> pathlib.Path:
    """CSV with all 6 labels in both rule and LLM columns, plus group columns."""
    labels = VALID_LABELS
    path = tmp_path / "combined.csv"
    pd.DataFrame({
        RULE_COL:         [labels[i % len(labels)] for i in range(n)],
        LLM_COL:          [labels[(i + 1) % len(labels)] for i in range(n)],
        "prompt_family":  [f"family_{i % 3}" for i in range(n)],
        "clarity_level":  [f"level_{i % 2}" for i in range(n)],
        "benchmark":      [f"bench_{i % 2}" for i in range(n)],
        "model_output":   [f"output {i}" for i in range(n)],
    }).to_csv(path, index=False)
    return path


# ── disagreement matrix ────────────────────────────────────────────────────

def test_compare_creates_disagreement_matrix_csv(tmp_path):
    csv_path = _make_combined_csv(tmp_path)
    compare(csv_path, tmp_path / "out")
    assert (tmp_path / "out" / "rule_vs_llm_disagreement_matrix.csv").exists()


def test_disagreement_matrix_is_6x6(tmp_path):
    csv_path = _make_combined_csv(tmp_path)
    out_dir = tmp_path / "out"
    compare(csv_path, out_dir)
    mat = pd.read_csv(out_dir / "rule_vs_llm_disagreement_matrix.csv", index_col=0)
    assert mat.shape == (len(VALID_LABELS), len(VALID_LABELS))


def test_disagreement_matrix_counts_are_non_negative(tmp_path):
    csv_path = _make_combined_csv(tmp_path)
    out_dir = tmp_path / "out"
    compare(csv_path, out_dir)
    mat = pd.read_csv(out_dir / "rule_vs_llm_disagreement_matrix.csv", index_col=0)
    assert (mat.values >= 0).all()


def test_compare_creates_heatmap_png(tmp_path):
    csv_path = _make_combined_csv(tmp_path)
    compare(csv_path, tmp_path / "out")
    assert (tmp_path / "out" / "rule_vs_llm_heatmap.png").exists()


# ── bar charts ─────────────────────────────────────────────────────────────

def test_compare_creates_llm_bar_chart_by_prompt_family(tmp_path):
    compare(_make_combined_csv(tmp_path), tmp_path / "out")
    assert (tmp_path / "out" / "policy_dist_llm_by_prompt_family.png").exists()


def test_compare_creates_rule_bar_chart_by_prompt_family(tmp_path):
    compare(_make_combined_csv(tmp_path), tmp_path / "out")
    assert (tmp_path / "out" / "policy_dist_rule_by_prompt_family.png").exists()


def test_compare_creates_llm_bar_chart_by_clarity_level(tmp_path):
    compare(_make_combined_csv(tmp_path), tmp_path / "out")
    assert (tmp_path / "out" / "policy_dist_llm_by_clarity_level.png").exists()


def test_compare_creates_llm_bar_chart_by_benchmark(tmp_path):
    compare(_make_combined_csv(tmp_path), tmp_path / "out")
    assert (tmp_path / "out" / "policy_dist_llm_by_benchmark.png").exists()


# ── family × clarity cross-tab ─────────────────────────────────────────────

def test_compare_creates_family_clarity_csv(tmp_path):
    compare(_make_combined_csv(tmp_path), tmp_path / "out")
    assert (tmp_path / "out" / "family_clarity_llm_policy.csv").exists()


def test_family_clarity_csv_has_required_columns(tmp_path):
    out_dir = tmp_path / "out"
    compare(_make_combined_csv(tmp_path), out_dir)
    ct = pd.read_csv(out_dir / "family_clarity_llm_policy.csv")
    assert "prompt_family" in ct.columns
    assert "clarity_level" in ct.columns
    assert LLM_COL in ct.columns
    assert "count" in ct.columns


def test_family_clarity_csv_counts_are_positive(tmp_path):
    out_dir = tmp_path / "out"
    compare(_make_combined_csv(tmp_path), out_dir)
    ct = pd.read_csv(out_dir / "family_clarity_llm_policy.csv")
    assert (ct["count"] > 0).all()


# ── graceful handling of missing columns ──────────────────────────────────

def test_compare_no_rule_col_does_not_crash(tmp_path):
    """If classified_policy is absent, compare() should not raise."""
    path = tmp_path / "no_rule.csv"
    pd.DataFrame({
        LLM_COL:        ["compliance", "refusal"] * 5,
        "prompt_family": ["fam_a", "fam_b"] * 5,
        "model_output":  [f"o{i}" for i in range(10)],
    }).to_csv(path, index=False)
    compare(path, tmp_path / "out")  # must not raise


def test_compare_no_rule_col_skips_disagreement_matrix(tmp_path):
    path = tmp_path / "no_rule.csv"
    pd.DataFrame({
        LLM_COL: ["compliance", "refusal"] * 5,
        "prompt_family": ["fam_a", "fam_b"] * 5,
    }).to_csv(path, index=False)
    out_dir = tmp_path / "out"
    compare(path, out_dir)
    assert not (out_dir / "rule_vs_llm_disagreement_matrix.csv").exists()


def test_compare_no_llm_col_does_not_crash(tmp_path):
    """If llm_policy_label is absent, compare() should not raise."""
    path = tmp_path / "no_llm.csv"
    pd.DataFrame({
        RULE_COL:        ["compliance", "refusal"] * 5,
        "prompt_family":  ["fam_a", "fam_b"] * 5,
    }).to_csv(path, index=False)
    compare(path, tmp_path / "out")  # must not raise


def test_compare_no_llm_col_skips_llm_bar_charts(tmp_path):
    path = tmp_path / "no_llm.csv"
    pd.DataFrame({
        RULE_COL:       ["compliance", "refusal"] * 5,
        "prompt_family": ["fam_a", "fam_b"] * 5,
    }).to_csv(path, index=False)
    out_dir = tmp_path / "out"
    compare(path, out_dir)
    assert not (out_dir / "policy_dist_llm_by_prompt_family.png").exists()


# ── output directory creation ──────────────────────────────────────────────

def test_compare_creates_nested_output_dir_if_missing(tmp_path):
    out_dir = tmp_path / "deep" / "nested" / "dir"
    assert not out_dir.exists()
    compare(_make_combined_csv(tmp_path), out_dir)
    assert out_dir.exists()


# ── groupby columns optional ───────────────────────────────────────────────

def test_compare_without_groupby_columns_does_not_crash(tmp_path):
    """CSV with only model output and labels — no prompt_family/clarity_level/benchmark."""
    path = tmp_path / "minimal.csv"
    pd.DataFrame({
        RULE_COL:   ["compliance", "refusal"] * 3,
        LLM_COL:    ["refusal", "compliance"] * 3,
        "model_output": [f"o{i}" for i in range(6)],
    }).to_csv(path, index=False)
    compare(path, tmp_path / "out")  # must not raise
