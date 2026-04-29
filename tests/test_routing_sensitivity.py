"""tests/test_routing_sensitivity.py — Milestone 5 tests."""

import numpy as np
import pandas as pd
import pytest

from mining.routing_sensitivity import (
    compute_routing_sensitivity,
    get_routing_sensitive_items,
    build_routing_sensitive_report,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

def _make_routing_df():
    """
    item_A: 4 families, all different clusters + high length variance → highest disagreement
    item_B: 4 families, same cluster + uniform lengths → zero disagreement
    item_C: 2 families, different clusters, but uniform lengths → lower than A
    """
    rows = [
        # item_A — max diversity: 4 distinct clusters + wide length spread
        {"item_id": "item_A", "prompt_family": "Refuse-first",    "char_count": 50},
        {"item_id": "item_A", "prompt_family": "Clarify-first",   "char_count": 900},
        {"item_id": "item_A", "prompt_family": "Helpful-baseline","char_count": 200},
        {"item_id": "item_A", "prompt_family": "Hierarchy-first", "char_count": 700},
        # item_B — all same cluster, uniform lengths
        {"item_id": "item_B", "prompt_family": "Refuse-first",    "char_count": 150},
        {"item_id": "item_B", "prompt_family": "Clarify-first",   "char_count": 150},
        {"item_id": "item_B", "prompt_family": "Helpful-baseline","char_count": 150},
        {"item_id": "item_B", "prompt_family": "Hierarchy-first", "char_count": 150},
        # item_C — 2 families, different clusters but uniform lengths
        {"item_id": "item_C", "prompt_family": "Refuse-first",    "char_count": 100},
        {"item_id": "item_C", "prompt_family": "Helpful-baseline","char_count": 100},
    ]
    df = pd.DataFrame(rows)
    # Cluster labels: A gets 0,1,2,3 (max diversity); B gets all 0; C gets 0,2
    cluster_labels = np.array([0, 1, 2, 3,   0, 0, 0, 0,   0, 2])
    return df, cluster_labels


# ── known-high item ranks first ───────────────────────────────────────────────

def test_high_disagreement_item_ranks_first():
    df, labels = _make_routing_df()
    result = compute_routing_sensitivity(df, labels)
    assert result.iloc[0]["item_id"] == "item_A"


def test_identical_outputs_rank_low():
    df, labels = _make_routing_df()
    result = compute_routing_sensitivity(df, labels)
    item_b = result[result["item_id"] == "item_B"].iloc[0]
    assert item_b["cluster_entropy"] == 0.0
    assert item_b["disagreement_score"] <= 0.1


# ── n_families correct ────────────────────────────────────────────────────────

def test_n_families_counted_correctly():
    df, labels = _make_routing_df()
    result = compute_routing_sensitivity(df, labels)
    assert result[result["item_id"] == "item_A"]["n_families"].iloc[0] == 4
    assert result[result["item_id"] == "item_C"]["n_families"].iloc[0] == 2


# ── grouping is correct ───────────────────────────────────────────────────────

def test_every_item_has_one_row_in_result():
    df, labels = _make_routing_df()
    result = compute_routing_sensitivity(df, labels)
    assert len(result) == df["item_id"].nunique()


# ── tied cases handled ────────────────────────────────────────────────────────

def test_tied_scores_broken_by_item_id():
    """Two items with identical scores should be sorted alphabetically."""
    df = pd.DataFrame([
        {"item_id": "beta",  "prompt_family": "A", "char_count": 100},
        {"item_id": "beta",  "prompt_family": "B", "char_count": 100},
        {"item_id": "alpha", "prompt_family": "A", "char_count": 100},
        {"item_id": "alpha", "prompt_family": "B", "char_count": 100},
    ])
    labels = np.array([0, 0, 0, 0])
    result = compute_routing_sensitivity(df, labels)
    # Both score 0; alpha < beta alphabetically → alpha should appear first
    assert result.iloc[0]["item_id"] == "alpha"


# ── invalid inputs ────────────────────────────────────────────────────────────

def test_missing_item_col_raises():
    df = pd.DataFrame({"prompt_family": ["A"], "char_count": [10]})
    with pytest.raises(ValueError, match="item_col"):
        compute_routing_sensitivity(df, np.array([0]))


def test_missing_family_col_raises():
    df = pd.DataFrame({"item_id": ["x"], "char_count": [10]})
    with pytest.raises(ValueError, match="family_col"):
        compute_routing_sensitivity(df, np.array([0]))


def test_label_length_mismatch_raises():
    df, _ = _make_routing_df()
    with pytest.raises(ValueError, match="labels"):
        compute_routing_sensitivity(df, np.array([0, 1]))


# ── get_routing_sensitive_items ───────────────────────────────────────────────

def test_get_routing_sensitive_items_top_n():
    df, labels = _make_routing_df()
    result = compute_routing_sensitivity(df, labels)
    top = get_routing_sensitive_items(result, top_n=1)
    assert len(top) == 1
    assert top.iloc[0]["item_id"] == "item_A"


def test_get_routing_sensitive_min_families_filter():
    df, labels = _make_routing_df()
    result = compute_routing_sensitivity(df, labels)
    top = get_routing_sensitive_items(result, top_n=10, min_families=4)
    # Only item_A and item_B have 4 families
    assert all(top["n_families"] >= 4)


# ── build_routing_sensitive_report ─────────────────────────────────────────────

def test_routing_sensitive_report_contains_top_items():
    df, labels = _make_routing_df()
    sensitivity = compute_routing_sensitivity(df, labels)
    report = build_routing_sensitive_report(df, sensitivity, labels, top_n=2)
    top_ids = sensitivity.head(2)["item_id"].tolist()
    assert set(report["item_id"].unique()).issubset(set(top_ids))


def test_routing_sensitive_report_has_cluster_column():
    df, labels = _make_routing_df()
    sensitivity = compute_routing_sensitivity(df, labels)
    report = build_routing_sensitive_report(df, sensitivity, labels, top_n=3)
    assert "cluster" in report.columns
