"""
audit/analysis.py
=================
Stage F — Primary routing analysis (PRD v4.1).

Functions for analysing audited policy labels to test whether prompt family
predicts response policy within the same item.

All functions accept a *labels DataFrame* with at minimum:
    * ``item_id``           — identifies the underlying item
    * ``prompt_family``     — the prompt controller used
    * ``primary_policy_label`` — the human-audited policy label

Public API
----------
    compute_routing_contingency(df)   → pd.DataFrame (families × labels)
    compute_within_item_switch_rate(df) → float in [0, 1]
    compute_policy_entropy(df)        → pd.Series indexed by item_id
    compute_cohens_kappa(a, b)        → float in [-1, 1]
    compute_label_distribution(df)    → pd.Series (label proportions)
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _entropy_bits(series: pd.Series) -> float:
    """Shannon entropy in bits for a categorical series."""
    counts = series.value_counts(dropna=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts.values / counts.values.sum()
    return float(scipy_entropy(probs, base=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_routing_contingency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a prompt_family × primary_policy_label contingency table.

    Each cell is the raw count of responses with that (family, label) pair.

    Parameters
    ----------
    df:
        Audit labels DataFrame with ``prompt_family`` and
        ``primary_policy_label`` columns.

    Returns
    -------
    pd.DataFrame
        Index = prompt families, columns = policy labels.
    """
    _require_cols(df, ["prompt_family", "primary_policy_label"])
    ct = pd.crosstab(df["prompt_family"], df["primary_policy_label"])
    return ct


def compute_within_item_switch_rate(df: pd.DataFrame) -> float:
    """
    Fraction of items that receive at least two distinct policy labels
    across prompt families.

    This is the primary evidence for H2 (prompt-routing hypothesis).

    Returns
    -------
    float
        Value in [0, 1].  0 means every item got the same label regardless
        of prompt family; 1 means every item had at least one policy switch.
    """
    _require_cols(df, ["item_id", "primary_policy_label"])
    per_item = df.groupby("item_id")["primary_policy_label"].nunique()
    return float((per_item > 1).mean())


def compute_policy_entropy(df: pd.DataFrame) -> pd.Series:
    """
    Per-item Shannon entropy (in bits) of the policy label distribution
    across prompt families.

    Higher entropy = more routing variability for that item.

    Returns
    -------
    pd.Series
        Indexed by ``item_id``, values are entropy in bits (≥ 0).
    """
    _require_cols(df, ["item_id", "primary_policy_label"])
    return (
        df.groupby("item_id")["primary_policy_label"]
        .apply(_entropy_bits)
        .rename("policy_entropy")
    )


def compute_cohens_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> float:
    """
    Cohen's kappa between two annotators given parallel label sequences.

    Parameters
    ----------
    labels_a, labels_b:
        Sequences of categorical labels of equal length.

    Returns
    -------
    float
        Kappa in [-1, 1].

    Raises
    ------
    ValueError
        If the two sequences have different lengths.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"labels_a and labels_b must have the same length "
            f"(got {len(labels_a)} and {len(labels_b)})."
        )
    a = np.array(labels_a)
    b = np.array(labels_b)
    n = len(a)

    # Observed agreement
    p_o = float(np.mean(a == b))

    # Expected agreement
    categories = np.unique(np.concatenate([a, b]))
    p_e = sum(
        (np.mean(a == cat) * np.mean(b == cat)) for cat in categories
    )

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def compute_label_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return the normalised frequency distribution of ``primary_policy_label``.

    Returns
    -------
    pd.Series
        Indexed by label, values sum to 1.0.
    """
    _require_cols(df, ["primary_policy_label"])
    counts = df["primary_policy_label"].value_counts(dropna=True)
    return counts / counts.sum()


def compute_family_effects_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each prompt family, compute the proportion of each policy label.

    Returns
    -------
    pd.DataFrame
        Index = prompt families, columns = policy labels, values = proportions
        (each row sums to 1.0).
    """
    _require_cols(df, ["prompt_family", "primary_policy_label"])
    ct = compute_routing_contingency(df)
    return ct.div(ct.sum(axis=1), axis=0)


def compute_routing_summary(df: pd.DataFrame) -> dict:
    """
    Convenience wrapper returning the key routing-analysis statistics as a dict.

    Returns
    -------
    dict with keys:
        ``switch_rate``, ``mean_policy_entropy``, ``label_distribution``,
        ``contingency_table``.
    """
    return {
        "switch_rate": compute_within_item_switch_rate(df),
        "mean_policy_entropy": float(compute_policy_entropy(df).mean()),
        "label_distribution": compute_label_distribution(df).to_dict(),
        "contingency_table": compute_routing_contingency(df),
    }
