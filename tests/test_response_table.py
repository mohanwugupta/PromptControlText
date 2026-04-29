"""tests/test_response_table.py — Milestone 1 tests."""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from mining.response_table import (
    build_response_table,
    load_phase,
    UNIFIED_COLUMNS,
    _REQUIRED_SHARED,
)

P1 = Path("tests/fixtures/mining_phase1.csv")
P2 = Path("tests/fixtures/mining_phase2.csv")


# ── column contract ────────────────────────────────────────────────────────

def test_unified_columns_present():
    df = build_response_table(P1, P2)
    for col in UNIFIED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_phase_column_values():
    df = build_response_table(P1, P2)
    assert set(df["phase"].unique()) == {1, 2}


# ── row count preservation ─────────────────────────────────────────────────

def test_row_count_preserved():
    p1 = pd.read_csv(P1)
    p2 = pd.read_csv(P2)
    df = build_response_table(P1, P2)
    # No rows should be dropped when all model_outputs are non-empty
    assert len(df) == len(p1) + len(p2)


def test_empty_model_output_warned_and_dropped(tmp_path):
    import shutil
    dest = tmp_path / "p1_bad.csv"
    shutil.copy(P1, dest)
    bad = pd.read_csv(dest)
    bad.loc[0, "model_output"] = ""
    bad.to_csv(dest, index=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = load_phase(dest, phase=1)
        assert any("empty model_output" in str(warning.message) for warning in w)
    # Row with empty output should be gone
    assert len(df) == len(bad) - 1


# ── item ID and prompt metadata preserved ─────────────────────────────────

def test_item_ids_preserved():
    raw = pd.read_csv(P1)
    df = build_response_table(P1, P2)
    phase1 = df[df["phase"] == 1]
    assert set(phase1["item_id"].tolist()) == set(raw["item_id"].tolist())


def test_prompt_family_preserved():
    raw = pd.read_csv(P1)
    df = build_response_table(P1, P2)
    phase1 = df[df["phase"] == 1]
    assert set(phase1["prompt_family"].unique()) == set(raw["prompt_family"].unique())


def test_benchmark_preserved():
    df = build_response_table(P1, P2)
    assert "XSTest" in df["benchmark"].values or "HarmBench" in df["benchmark"].values


# ── missing file handling ──────────────────────────────────────────────────

def test_missing_file_raises_by_default():
    with pytest.raises(FileNotFoundError):
        build_response_table("nonexistent.csv", P2)


def test_missing_file_skipped_when_require_both_false(tmp_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = build_response_table("nonexistent.csv", P2, require_both=False)
        assert any("Skipping" in str(warning.message) for warning in w)
    assert len(df) > 0


# ── required columns enforcement ──────────────────────────────────────────

def test_missing_required_column_raises(tmp_path):
    bad = pd.read_csv(P1).drop(columns=["item_id"])
    dest = tmp_path / "bad.csv"
    bad.to_csv(dest, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_phase(dest, phase=1)


# ── phase 2 NaN-fill for phase-1-only scorer columns ──────────────────────

def test_phase2_scorer_cols_are_nan():
    df = build_response_table(P1, P2)
    phase2 = df[df["phase"] == 2]
    # refusal_score is phase-1 only; should be NaN for phase-2 rows
    assert phase2["refusal_score"].isna().all()


# ── load_phase ──────────────────────────────────────────────────────────────

def test_load_phase_returns_unified_schema():
    df = load_phase(P1, phase=1)
    for col in UNIFIED_COLUMNS:
        assert col in df.columns
    assert (df["phase"] == 1).all()
