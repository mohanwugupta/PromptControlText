"""tests/test_exemplar_selection.py — Milestone 4 tests."""

import numpy as np
import pandas as pd
import pytest

from mining.clustering import run_text_clustering
from mining.exemplars import (
    select_centroid_exemplars,
    select_random_exemplars,
    build_exemplar_table,
)


def _make_df(n_each: int = 20) -> pd.DataFrame:
    texts = (
        [f"I cannot help with request {i}. I will not assist." for i in range(n_each)]
        + [f"Sure, here is how to do task {i}. Step 1 start." for i in range(n_each)]
    )
    return pd.DataFrame({
        "model_output": texts,
        "item_id": [f"item_{i:03d}" for i in range(len(texts))],
        "benchmark": ["XSTest"] * n_each + ["HarmBench"] * n_each,
        "prompt_family": ["Refuse-first"] * n_each + ["Helpful-baseline"] * n_each,
        "gold_label": ["safe"] * n_each + ["unsafe"] * n_each,
    })


# ── centroid exemplars ───────────────────────────────────────────────────────

def test_centroid_exemplars_correct_clusters():
    df = _make_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    exemplars = select_centroid_exemplars(df, r, n_per_cluster=3)
    assert "cluster" in exemplars.columns
    assert set(exemplars["cluster"].unique()).issubset({0, 1})


def test_centroid_exemplars_n_per_cluster():
    df = _make_df(n_each=30)
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    exemplars = select_centroid_exemplars(df, r, n_per_cluster=3)
    for cid in [0, 1]:
        assert len(exemplars[exemplars["cluster"] == cid]) <= 3


def test_centroid_exemplars_no_duplicates_when_enough_data():
    df = _make_df(n_each=30)
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    exemplars = select_centroid_exemplars(df, r, n_per_cluster=3)
    # No duplicate row indices
    assert exemplars.index.is_unique or len(exemplars) == len(exemplars.drop_duplicates())


def test_centroid_exemplars_label_mismatch_raises():
    df = _make_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    short_df = df.iloc[:5]
    with pytest.raises(ValueError, match="labels"):
        select_centroid_exemplars(short_df, r, n_per_cluster=2)


# ── random exemplars ─────────────────────────────────────────────────────────

def test_random_exemplars_reproducible():
    df = _make_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    ex1 = select_random_exemplars(df, r, n_per_cluster=5, random_state=0)
    ex2 = select_random_exemplars(df, r, n_per_cluster=5, random_state=0)
    pd.testing.assert_frame_equal(ex1.reset_index(drop=True),
                                  ex2.reset_index(drop=True))


def test_random_exemplars_no_upsampling():
    """When a cluster has fewer rows than n_per_cluster, return all rows."""
    df = _make_df(n_each=5)   # only 5 rows per cluster
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    exemplars = select_random_exemplars(df, r, n_per_cluster=20, random_state=0)
    for cid in [0, 1]:
        cluster_rows = exemplars[exemplars["cluster"] == cid]
        max_possible = (r.labels == cid).sum()
        assert len(cluster_rows) <= max_possible


def test_random_exemplars_from_correct_clusters():
    """Each exemplar row's cluster column must match the clustering label
    for the corresponding model_output text."""
    df = _make_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    # Build a lookup: model_output → cluster label (assuming unique texts)
    text_to_cluster = {row["model_output"]: r.labels[i]
                       for i, row in df.iterrows()}
    exemplars = select_random_exemplars(df, r, n_per_cluster=4, random_state=99)
    for _, row in exemplars.iterrows():
        cid = int(row["cluster"])
        expected_cid = int(text_to_cluster[row["model_output"]])
        assert cid == expected_cid


# ── build_exemplar_table ─────────────────────────────────────────────────────

def test_build_exemplar_table_has_required_columns():
    df = _make_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    from mining.clustering import top_terms_per_cluster
    terms = top_terms_per_cluster(r)
    tbl = build_exemplar_table(df, r, terms)
    for col in ["cluster", "selection_method", "top_terms"]:
        assert col in tbl.columns


def test_build_exemplar_table_selection_methods():
    df = _make_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    from mining.clustering import top_terms_per_cluster
    terms = top_terms_per_cluster(r)
    tbl = build_exemplar_table(df, r, terms)
    assert set(tbl["selection_method"].unique()).issubset({"centroid", "random"})


def test_build_exemplar_table_top_terms_populated():
    df = _make_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    from mining.clustering import top_terms_per_cluster
    terms = top_terms_per_cluster(r)
    tbl = build_exemplar_table(df, r, terms)
    assert tbl["top_terms"].notna().all()
    assert (tbl["top_terms"].str.len() > 0).any()
