"""
mining/exemplars.py
===================
Milestone 4 — Exemplar selection (PRD v3 §13 M4).

Two selection strategies:
* **centroid** : rows closest (by cosine distance) to the KMeans centroid.
* **random**   : stratified random sample per cluster (audit set).

Both strategies are reproducible given a fixed ``random_state``.
Duplicate exemplars are avoided when the cluster has enough members.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from mining.clustering import ClusterResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_centroid_exemplars(
    df: pd.DataFrame,
    result: ClusterResult,
    n_per_cluster: int = 3,
) -> pd.DataFrame:
    """
    For each cluster, return the *n_per_cluster* rows closest to the centroid.

    Distance is measured in the TF-IDF (or hybrid) representation space.
    Ties are broken by row order.

    Returns a DataFrame subset of *df* with an added ``cluster`` column.
    """
    if len(df) != len(result.labels):
        raise ValueError(
            f"df has {len(df)} rows but result has {len(result.labels)} labels."
        )

    mat = result.tfidf_matrix
    X = mat.toarray() if issparse(mat) else np.array(mat)
    centers = result.kmeans.cluster_centers_

    rows = []
    for cid in range(result.n_clusters):
        mask = result.labels == cid
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        cluster_vecs = X[indices]
        centroid = centers[cid]

        # Cosine distances
        norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        cluster_normed = cluster_vecs / norms
        c_norm = np.linalg.norm(centroid)
        c_unit = centroid / (c_norm if c_norm > 0 else 1e-10)
        similarities = cluster_normed.dot(c_unit)
        top_k = min(n_per_cluster, len(indices))
        # Avoid duplicates by unique original indices
        top_idx = indices[np.argsort(similarities)[::-1][:top_k]]

        subset = df.iloc[top_idx].copy()
        subset["cluster"] = cid
        rows.append(subset)

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["cluster"])

    return pd.concat(rows, ignore_index=True)


def select_random_exemplars(
    df: pd.DataFrame,
    result: ClusterResult,
    n_per_cluster: int = 5,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified random sample of *n_per_cluster* rows per cluster.

    When a cluster has fewer rows than *n_per_cluster*, all rows are returned
    (no upsampling / duplication).
    """
    if len(df) != len(result.labels):
        raise ValueError(
            f"df has {len(df)} rows but result has {len(result.labels)} labels."
        )

    rng = np.random.default_rng(random_state)
    rows = []
    for cid in range(result.n_clusters):
        indices = np.where(result.labels == cid)[0]
        if len(indices) == 0:
            continue
        k = min(n_per_cluster, len(indices))
        chosen = rng.choice(indices, k, replace=False)
        subset = df.iloc[chosen].copy()
        subset["cluster"] = cid
        rows.append(subset)

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["cluster"])

    return pd.concat(rows, ignore_index=True)


def build_exemplar_table(
    df: pd.DataFrame,
    result: ClusterResult,
    top_terms: dict,
    n_centroid: int = 3,
    n_random: int = 5,
    *,
    text_col: str = "model_output",
    meta_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Combine centroid and random exemplars into one audit-ready table.

    Columns returned:
        cluster, selection_method, top_terms,
        <meta_cols>, <text_col>

    Parameters
    ----------
    top_terms : dict
        Output of ``clustering.top_terms_per_cluster()``.
    meta_cols : list[str] | None
        Columns from *df* to preserve alongside the text.
        Defaults to ``["item_id", "benchmark", "prompt_family", "gold_label"]``.
    """
    if meta_cols is None:
        meta_cols = ["item_id", "benchmark", "prompt_family", "gold_label"]

    keep = [c for c in meta_cols if c in df.columns]

    centroid_ex = select_centroid_exemplars(df, result, n_per_cluster=n_centroid)
    centroid_ex["selection_method"] = "centroid"

    random_ex = select_random_exemplars(
        df, result, n_per_cluster=n_random, random_state=random_state
    )
    random_ex["selection_method"] = "random"

    combined = pd.concat([centroid_ex, random_ex], ignore_index=True)
    combined = combined.drop_duplicates(subset=keep + ["selection_method"])
    combined["top_terms"] = combined["cluster"].map(
        lambda c: ", ".join(top_terms.get(int(c), []))
    )

    out_cols = ["cluster", "selection_method", "top_terms"] + keep
    if text_col in combined.columns:
        out_cols.append(text_col)

    return combined[[c for c in out_cols if c in combined.columns]].reset_index(drop=True)
