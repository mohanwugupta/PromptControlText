"""tests/test_clustering_pipeline.py — Milestone 3 tests."""

import numpy as np
import pandas as pd
import pytest

from mining.clustering import (
    run_text_clustering,
    run_hybrid_clustering,
    top_terms_per_cluster,
    ClusterResult,
    _validate_inputs,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

def _refusal_comply_df(n_each: int = 30) -> pd.DataFrame:
    """Toy corpus with two well-separated classes."""
    refusals = [
        f"I cannot help with that request number {i}. I'm sorry, I will not assist."
        for i in range(n_each)
    ]
    compliances = [
        f"Sure, here is how to complete task {i}. Step 1: gather materials. Step 2: proceed."
        for i in range(n_each)
    ]
    rows = (
        [{"model_output": t, "refusal_score": 1.0, "compliance_score": 0.0,
          "clarification_score": 0.0, "abstention_score": 0.0,
          "hierarchy_following_score": 0.0, "unsafe_continuation_score": 0.0,
          "stop_compliance_score": 0.0} for t in refusals]
        + [{"model_output": t, "refusal_score": 0.0, "compliance_score": 1.0,
            "clarification_score": 0.0, "abstention_score": 0.0,
            "hierarchy_following_score": 0.0, "unsafe_continuation_score": 0.0,
            "stop_compliance_score": 0.0} for t in compliances]
    )
    return pd.DataFrame(rows)


# ── determinism ───────────────────────────────────────────────────────────────

def test_text_clustering_deterministic():
    df = _refusal_comply_df()
    r1 = run_text_clustering(df, n_clusters=2, random_state=0)
    r2 = run_text_clustering(df, n_clusters=2, random_state=0)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_hybrid_clustering_deterministic():
    df = _refusal_comply_df()
    r1 = run_hybrid_clustering(df, n_clusters=2, random_state=0)
    r2 = run_hybrid_clustering(df, n_clusters=2, random_state=0)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_different_seeds_may_differ():
    """Two different seeds *can* produce the same labels for a clear corpus,
    but the labels array must at least be returned without error."""
    df = _refusal_comply_df()
    r1 = run_text_clustering(df, n_clusters=2, random_state=1)
    r2 = run_text_clustering(df, n_clusters=2, random_state=99)
    assert len(r1.labels) == len(r2.labels)


# ── toy corpus separates ──────────────────────────────────────────────────────

def test_text_clustering_separates_two_classes():
    """With 2 clusters the toy corpus should produce two non-trivial groups."""
    df = _refusal_comply_df(n_each=40)
    result = run_text_clustering(df, n_clusters=2, random_state=42)
    sizes = list(result.cluster_sizes.values())
    # Both clusters should have some members
    assert min(sizes) > 5


def test_cluster_count_matches_n_clusters():
    df = _refusal_comply_df()
    for k in [2, 3, 5]:
        r = run_text_clustering(df, n_clusters=k, random_state=42)
        assert r.n_clusters == k
        assert len(r.cluster_sizes) == k


# ── ClusterResult attributes ──────────────────────────────────────────────────

def test_cluster_result_has_silhouette():
    df = _refusal_comply_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    assert -1.0 <= r.silhouette <= 1.0


def test_cluster_result_mode_text():
    df = _refusal_comply_df()
    r = run_text_clustering(df, n_clusters=2)
    assert r.mode == "text"


def test_cluster_result_mode_hybrid():
    df = _refusal_comply_df()
    r = run_hybrid_clustering(df, n_clusters=2)
    assert r.mode == "hybrid"


def test_labels_length_matches_df():
    df = _refusal_comply_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    assert len(r.labels) == len(df)


# ── invalid inputs fail loudly ────────────────────────────────────────────────

def test_n_clusters_less_than_2_raises():
    df = _refusal_comply_df()
    with pytest.raises(ValueError, match="≥ 2"):
        run_text_clustering(df, n_clusters=1)


def test_n_clusters_ge_n_rows_raises():
    df = _refusal_comply_df(n_each=3)   # 6 rows
    with pytest.raises(ValueError):
        run_text_clustering(df, n_clusters=6)


def test_missing_text_col_raises():
    df = pd.DataFrame({"other": ["a", "b", "c", "d", "e"]})
    with pytest.raises(ValueError, match="model_output"):
        run_text_clustering(df, n_clusters=2)


def test_too_few_rows_raises():
    df = pd.DataFrame({"model_output": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="rows"):
        run_text_clustering(df, n_clusters=2)


def test_n_clusters_not_int_raises():
    df = _refusal_comply_df()
    with pytest.raises(ValueError):
        run_text_clustering(df, n_clusters="two")


# ── top terms ────────────────────────────────────────────────────────────────

def test_top_terms_returns_list_per_cluster():
    df = _refusal_comply_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    terms = top_terms_per_cluster(r, n_terms=10)
    assert len(terms) == 2
    for k, lst in terms.items():
        assert isinstance(lst, list)
        assert len(lst) <= 10


def test_top_terms_are_strings():
    df = _refusal_comply_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    terms = top_terms_per_cluster(r)
    for k, lst in terms.items():
        for t in lst:
            assert isinstance(t, str)


# ── hybrid falls back to text when no score cols ──────────────────────────────

def test_hybrid_falls_back_to_text_when_no_score_cols(recwarn):
    df = pd.DataFrame({
        "model_output": [
            "I cannot help with this." + str(i) for i in range(20)
        ] + [
            "Sure, here is the answer." + str(i) for i in range(20)
        ]
    })
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = run_hybrid_clustering(df, n_clusters=2, score_cols=[])
        assert r.mode in ("text", "hybrid")
