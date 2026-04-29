"""
mining/response_table.py
========================
Milestone 1 — Unified response table builder (PRD v3 §13).

Loads Phase 1 and Phase 2 CSV outputs into a single row-level DataFrame.
Each row is one model output paired with its prompt metadata.

Design constraints
------------------
* No rows are silently dropped.  Any row lost triggers a warning with a count.
* Required columns must all be present; missing ones raise ValueError.
* Phase 2 lacks the refusal/compliance scorer columns; they are back-filled
  with NaN so the unified schema is consistent.
* ``phase`` column (1 or 2) is added to preserve provenance.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------
_REQUIRED_SHARED = [
    "item_id",
    "benchmark",
    "prompt_family",
    "model_output",
    "gold_label",
]

_SCORE_COLS_PHASE1 = [
    "refusal_score",
    "compliance_score",
    "clarification_score",
    "abstention_score",
    "hierarchy_following_score",
    "unsafe_continuation_score",
    "stop_compliance_score",
]

_SCORE_COLS_PHASE2 = [
    "hierarchy_following_score",
    "unsafe_continuation_score",
    "stop_compliance_score",
]

# Full unified schema — cols absent in a phase are NaN-filled.
UNIFIED_COLUMNS = [
    "item_id",
    "benchmark",
    "domain",
    "input_text",
    "gold_label",
    "prompt_family",
    "prompt_variant",
    "clarity_level",
    "context_condition",
    "ambiguity_level",
    "model_output",
    "score",
    "policy_label",
    "refusal_score",
    "compliance_score",
    "clarification_score",
    "abstention_score",
    "hierarchy_following_score",
    "unsafe_continuation_score",
    "stop_compliance_score",
    "phase",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_required(df: pd.DataFrame, path: str, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def _add_missing_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add any absent columns as NaN so the schema is uniform."""
    for col in cols:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def _load_one(path: Union[str, Path], phase: int) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    before = None
    df = pd.read_csv(path)
    before = len(df)

    _check_required(df, str(path), _REQUIRED_SHARED)

    # Drop rows with no model output — these are unusable for mining.
    missing_output = df["model_output"].isna() | (df["model_output"].astype(str).str.strip() == "")
    n_dropped = missing_output.sum()
    if n_dropped > 0:
        warnings.warn(
            f"{path.name}: dropping {n_dropped}/{before} rows with empty model_output",
            stacklevel=3,
        )
    df = df[~missing_output].copy()

    df["phase"] = phase
    df = _add_missing_cols(df, UNIFIED_COLUMNS)

    return df[UNIFIED_COLUMNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_response_table(
    phase1_path: Union[str, Path],
    phase2_path: Union[str, Path],
    *,
    require_both: bool = True,
) -> pd.DataFrame:
    """
    Load Phase 1 and Phase 2 CSV files into one unified DataFrame.

    Parameters
    ----------
    phase1_path, phase2_path : path-like
        Paths to the respective results CSVs.
    require_both : bool
        If True (default), both files must exist.  If False, a missing file
        is skipped with a warning and only the available phase is returned.

    Returns
    -------
    pd.DataFrame
        Unified mining table with columns defined by ``UNIFIED_COLUMNS``.
    """
    frames = []

    for path, phase in [(phase1_path, 1), (phase2_path, 2)]:
        path = Path(path)
        if not path.exists():
            if require_both:
                raise FileNotFoundError(f"Results file not found: {path}")
            warnings.warn(f"Skipping missing file: {path}", stacklevel=2)
            continue
        frames.append(_load_one(path, phase))

    if not frames:
        raise ValueError("No result files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_phase(
    path: Union[str, Path],
    phase: int,
) -> pd.DataFrame:
    """
    Load a single phase CSV into the unified schema.
    Useful for incremental / streaming workflows.
    """
    return _load_one(path, phase)
