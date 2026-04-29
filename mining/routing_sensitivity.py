"""
mining/routing_sensitivity.py
==============================
Milestone 5 — Routing-sensitive item detection (PRD v3 §11 Analysis 4).

An item is *routing-sensitive* when the same underlying prompt elicits
meaningfully different policy-like responses across different prompt families.

Disagreement is measured three ways and combined into a single
``disagreement_score`` (0 … 1):

1. **cluster entropy**      — normalised Shannon entropy of cluster label
                              distribution across prompt families.
2. **policy classifier**    — normalised entropy of ``policy_label`` values
                              if present (falls back to 0 when absent).
3. **length variance**      — coefficient of variation of ``char_count``
                              across prompt families (normalised to [0,1]).

The three components are averaged; items with higher scores showed more
varied routing under different controllers.

Ties are broken alphabetically by ``item_id``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalised_entropy(values: pd.Series) -> float:
    """Return H / log2(n_unique) so the result is in [0, 1]."""
    counts = values.value_counts(dropna=True)
    if len(counts) <= 1:
        return 0.0
    probs = counts.values / counts.values.sum()
    h = float(scipy_entropy(probs, base=2))
    max_h = float(np.log2(len(counts)))
    return h / max_h if max_h > 0 else 0.0


def _cv_normalised(values: pd.Series) -> float:
    """Coefficient of variation clamped to [0, 1]."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    mean = numeric.mean()
    if mean == 0:
        return 0.0
    cv = numeric.std() / abs(mean)
    return float(min(cv, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_routing_sensitivity(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    item_col: str = "item_id",
    family_col: str = "prompt_family",
    policy_col: Optional[str] = "policy_label",
    char_col: Optional[str] = "char_count",
    weights: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute a per-item routing disagreement score.

    Parameters
    ----------
    df : DataFrame
        Mining table (one row per output).
    cluster_labels : np.ndarray
        Cluster label for each row (output of ``ClusterResult.labels``).
    item_col : str
        Column identifying the underlying item (independent of prompt).
    family_col : str
        Column identifying the prompt controller family.
    policy_col : str | None
        Column with policy classifier labels.  If absent, that component
        contributes 0.
    char_col : str | None
        Column with character count.  If absent, that component contributes 0.
    weights : dict | None
        Keys: "cluster", "policy", "length".  Default: equal thirds.

    Returns
    -------
    pd.DataFrame with columns:
        item_id, n_outputs, n_families,
        cluster_entropy, policy_entropy, length_cv,
        disagreement_score
    Sorted descending by disagreement_score, then item_id.
    """
    if item_col not in df.columns:
        raise ValueError(f"item_col '{item_col}' not found in DataFrame.")
    if family_col not in df.columns:
        raise ValueError(f"family_col '{family_col}' not found in DataFrame.")
    if len(df) != len(cluster_labels):
        raise ValueError(
            f"df has {len(df)} rows but cluster_labels has {len(cluster_labels)}."
        )

    if weights is None:
        weights = {"cluster": 1 / 3, "policy": 1 / 3, "length": 1 / 3}

    work = df[[c for c in [item_col, family_col, policy_col, char_col] if c and c in df.columns]].copy()
    work = work.reset_index(drop=True)
    work["_cluster"] = cluster_labels

    records = []
    for item_id, grp in work.groupby(item_col):
        n_outputs = len(grp)
        n_families = grp[family_col].nunique() if family_col in grp.columns else 1

        # Component 1: cluster entropy
        cluster_ent = _normalised_entropy(grp["_cluster"])

        # Component 2: policy entropy
        policy_ent = 0.0
        if policy_col and policy_col in grp.columns:
            valid = grp[policy_col].dropna()
            if len(valid) > 0:
                policy_ent = _normalised_entropy(valid)

        # Component 3: length CV
        length_cv = 0.0
        if char_col and char_col in grp.columns:
            length_cv = _cv_normalised(grp[char_col])

        score = (
            weights["cluster"] * cluster_ent
            + weights["policy"] * policy_ent
            + weights["length"] * length_cv
        )

        records.append({
            item_col: item_id,
            "n_outputs": n_outputs,
            "n_families": n_families,
            "cluster_entropy": round(cluster_ent, 4),
            "policy_entropy": round(policy_ent, 4),
            "length_cv": round(length_cv, 4),
            "disagreement_score": round(score, 4),
        })

    result = pd.DataFrame(records)
    result = result.sort_values(
        ["disagreement_score", item_col], ascending=[False, True]
    ).reset_index(drop=True)
    return result


def get_routing_sensitive_items(
    sensitivity_df: pd.DataFrame,
    top_n: int = 50,
    min_families: int = 2,
) -> pd.DataFrame:
    """
    Filter to the *top_n* most routing-sensitive items that were seen under
    at least *min_families* distinct prompt families.
    """
    filtered = sensitivity_df[sensitivity_df["n_families"] >= min_families]
    return filtered.head(top_n).reset_index(drop=True)


def build_routing_sensitive_report(
    df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    top_n: int = 20,
    *,
    item_col: str = "item_id",
    text_col: str = "model_output",
    family_col: str = "prompt_family",
) -> pd.DataFrame:
    """
    For each of the top-N routing-sensitive items, gather their outputs
    across prompt families with cluster assignments.

    Returns a long-format DataFrame for inspection / report export.
    """
    top_items = sensitivity_df.head(top_n)[item_col].tolist()

    work = df.copy().reset_index(drop=True)
    work["_cluster"] = cluster_labels
    work = work[work[item_col].isin(top_items)]

    keep_cols = [c for c in [item_col, "benchmark", family_col, text_col, "_cluster",
                              "disagreement_score"] if c in work.columns]

    # Merge disagreement scores
    work = work.merge(
        sensitivity_df[[item_col, "disagreement_score", "n_families",
                        "cluster_entropy"]],
        on=item_col,
        how="left",
    )

    final_cols = [c for c in keep_cols + ["disagreement_score", "n_families",
                                           "cluster_entropy"] if c in work.columns]
    # Deduplicate column names introduced by merge
    final_cols = list(dict.fromkeys(final_cols))

    return work[final_cols].rename(columns={"_cluster": "cluster"}).reset_index(drop=True)
