"""
mining/clustering.py
====================
Milestone 3 — Cluster pipeline (PRD v3 §11, §15).

Two clustering modes:
* **text-only**  : TF-IDF on raw ``model_output`` text → KMeans
* **hybrid**     : TF-IDF + numeric behavior scores → KMeans

Design choices (per PRD v3 §15):
* TF-IDF is the primary text representation (interpretable, deterministic).
* Dimensionality reduction (UMAP/TSNE) is used only for *visualisation*,
  not for the main cluster assignment.
* Cluster count is not fixed in advance; callers pass ``n_clusters``.
* Deterministic seeds: passing ``random_state`` produces reproducible labels.
* The pipeline fails loudly on malformed inputs.

Public API
----------
ClusterResult  — dataclass holding labels, model objects, diagnostics
run_text_clustering(df, n_clusters, random_state) -> ClusterResult
run_hybrid_clustering(df, n_clusters, random_state) -> ClusterResult
top_terms_per_cluster(result, n_terms) -> dict[int, list[str]]
reduce_for_viz(result, method) -> np.ndarray  (2-D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, issparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_ROWS = 4            # refuse clustering on tiny data
_DEFAULT_MAX_FEATURES = 5_000
_DEFAULT_NGRAM = (1, 2)

_HYBRID_SCORE_COLS = [
    "refusal_score",
    "compliance_score",
    "clarification_score",
    "abstention_score",
    "hierarchy_following_score",
    "unsafe_continuation_score",
    "stop_compliance_score",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    """Container returned by every clustering run."""
    labels: np.ndarray                     # shape (n_rows,)  int cluster IDs
    n_clusters: int
    mode: str                              # "text" | "hybrid"
    tfidf_matrix: object                   # sparse or dense matrix used
    tfidf_vectorizer: TfidfVectorizer
    kmeans: KMeans
    silhouette: float                      # −1 … 1
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    # 2D coords populated lazily by reduce_for_viz
    coords_2d: Optional[np.ndarray] = None

    def __post_init__(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        self.cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_tfidf(texts: pd.Series, max_features: int, ngram_range: tuple) -> tuple:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    mat = vec.fit_transform(texts.astype(str).fillna(""))
    return vec, mat


def _run_kmeans(matrix, n_clusters: int, random_state: int) -> KMeans:
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    km.fit(matrix)
    return km


def _safe_silhouette(matrix, labels: np.ndarray) -> float:
    """Return silhouette or -1 if it cannot be computed."""
    try:
        X = matrix.toarray() if issparse(matrix) else matrix
        return float(silhouette_score(X, labels, sample_size=min(5000, len(labels))))
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_text_clustering(
    df: pd.DataFrame,
    n_clusters: int,
    *,
    text_col: str = "model_output",
    max_features: int = _DEFAULT_MAX_FEATURES,
    ngram_range: tuple = _DEFAULT_NGRAM,
    random_state: int = 42,
) -> ClusterResult:
    """
    Cluster model outputs using TF-IDF representation only.

    Parameters
    ----------
    df : DataFrame
        Must contain *text_col*.
    n_clusters : int
        Number of clusters.  Must be ≥ 2 and < len(df).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    ClusterResult
    """
    _validate_inputs(df, n_clusters, text_col)

    vec, mat = _build_tfidf(df[text_col], max_features, ngram_range)
    km = _run_kmeans(mat, n_clusters, random_state)
    labels = km.labels_
    sil = _safe_silhouette(mat, labels)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        mode="text",
        tfidf_matrix=mat,
        tfidf_vectorizer=vec,
        kmeans=km,
        silhouette=sil,
    )


def run_hybrid_clustering(
    df: pd.DataFrame,
    n_clusters: int,
    *,
    text_col: str = "model_output",
    score_cols: Optional[List[str]] = None,
    max_features: int = _DEFAULT_MAX_FEATURES,
    ngram_range: tuple = _DEFAULT_NGRAM,
    random_state: int = 42,
) -> ClusterResult:
    """
    Cluster using TF-IDF text features concatenated with numeric score columns.

    Score columns absent from *df* are silently filled with 0.0.
    """
    _validate_inputs(df, n_clusters, text_col)

    if score_cols is None:
        score_cols = [c for c in _HYBRID_SCORE_COLS if c in df.columns]
    if not score_cols:
        import warnings
        warnings.warn(
            "No score columns found; hybrid clustering falls back to text-only.",
            stacklevel=2,
        )
        return run_text_clustering(
            df, n_clusters,
            text_col=text_col,
            max_features=max_features,
            ngram_range=ngram_range,
            random_state=random_state,
        )

    vec, tfidf_mat = _build_tfidf(df[text_col], max_features, ngram_range)

    # Scale numeric scores
    score_data = df[score_cols].fillna(0.0).values.astype(float)
    scaler = StandardScaler()
    score_scaled = scaler.fit_transform(score_data)

    from scipy.sparse import csr_matrix
    combined = hstack([tfidf_mat, csr_matrix(score_scaled)])

    km = _run_kmeans(combined, n_clusters, random_state)
    labels = km.labels_
    sil = _safe_silhouette(combined, labels)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        mode="hybrid",
        tfidf_matrix=combined,
        tfidf_vectorizer=vec,
        kmeans=km,
        silhouette=sil,
    )


def top_terms_per_cluster(
    result: ClusterResult,
    n_terms: int = 15,
) -> Dict[int, List[str]]:
    """
    Return the top TF-IDF terms per cluster using centroid weights.

    Only works for the TF-IDF portion of the matrix.
    For hybrid results this uses the TF-IDF block (left columns).
    """
    vocab = result.tfidf_vectorizer.get_feature_names_out()
    n_vocab = len(vocab)

    # centroids may be from a sparse combined matrix — take first n_vocab cols
    centers = result.kmeans.cluster_centers_[:, :n_vocab]

    top: Dict[int, List[str]] = {}
    for cid in range(result.n_clusters):
        indices = np.argsort(centers[cid])[::-1][:n_terms]
        top[cid] = [vocab[i] for i in indices]
    return top


def reduce_for_viz(
    result: ClusterResult,
    method: str = "tsne",
    random_state: int = 42,
    n_sample: int = 5_000,
) -> np.ndarray:
    """
    Return 2-D coordinates for visualisation.  Stores in result.coords_2d.

    Parameters
    ----------
    method : "tsne" | "umap"
    n_sample : int
        Sub-sample size for large corpora.
    """
    mat = result.tfidf_matrix
    X = mat.toarray() if issparse(mat) else mat
    n = X.shape[0]
    if n > n_sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, n_sample, replace=False)
        X = X[idx]

    if method == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=min(30, n_sample // 4),
            n_iter=500,
        ).fit_transform(X)
    elif method == "umap":
        try:
            import umap  # type: ignore
            coords = umap.UMAP(
                n_components=2, random_state=random_state
            ).fit_transform(X)
        except ImportError as exc:
            raise ImportError("Install `umap-learn` to use method='umap'") from exc
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'tsne' or 'umap'.")

    result.coords_2d = coords
    return coords


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_inputs(df: pd.DataFrame, n_clusters: int, text_col: str) -> None:
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    if len(df) < _MIN_ROWS:
        raise ValueError(
            f"DataFrame has only {len(df)} rows; need at least {_MIN_ROWS}."
        )
    if not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError(f"n_clusters must be an integer ≥ 2, got {n_clusters!r}.")
    if n_clusters >= len(df):
        raise ValueError(
            f"n_clusters ({n_clusters}) must be < number of rows ({len(df)})."
        )
