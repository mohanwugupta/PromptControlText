"""tests/test_cluster_enrichment.py — enrichment aggregation tests."""

import numpy as np
import pandas as pd
import pytest

from mining.reports import plot_enrichment_heatmap
import matplotlib
matplotlib.use("Agg")


def _make_enrichment_df():
    """
    3 families × 40 rows each = 120 rows.
    Family A → mostly cluster 0
    Family B → mostly cluster 1
    Family C → mixed
    """
    rng = np.random.default_rng(7)
    families = []
    labels = []
    for fam, probs in [("FamA", [0.9, 0.1]), ("FamB", [0.1, 0.9]), ("FamC", [0.5, 0.5])]:
        n = 40
        families.extend([fam] * n)
        labels.extend(rng.choice([0, 1], size=n, p=probs).tolist())
    return pd.DataFrame({"prompt_family": families}), np.array(labels)


# ── enrichment table aggregates correctly ────────────────────────────────────

def test_enrichment_table_shape(tmp_path):
    df, labels = _make_enrichment_df()
    out = tmp_path / "enrich.png"
    ct = plot_enrichment_heatmap(df, labels, out)
    # 3 families × 2 clusters
    assert ct.shape == (3, 2)


def test_enrichment_row_sums_to_one(tmp_path):
    df, labels = _make_enrichment_df()
    out = tmp_path / "enrich.png"
    ct = plot_enrichment_heatmap(df, labels, out)
    for fam in ct.index:
        assert abs(ct.loc[fam].sum() - 1.0) < 1e-6, f"Row {fam} does not sum to 1"


def test_enrichment_values_in_0_1(tmp_path):
    df, labels = _make_enrichment_df()
    out = tmp_path / "enrich.png"
    ct = plot_enrichment_heatmap(df, labels, out)
    assert (ct.values >= 0).all()
    assert (ct.values <= 1.0 + 1e-9).all()


def test_enrichment_high_for_dominant_family(tmp_path):
    """FamA is dominated by cluster 0 → proportion > 0.7."""
    df, labels = _make_enrichment_df()
    out = tmp_path / "enrich.png"
    ct = plot_enrichment_heatmap(df, labels, out)
    assert ct.loc["FamA", 0] > 0.7


def test_enrichment_figure_written(tmp_path):
    df, labels = _make_enrichment_df()
    out = tmp_path / "enrich.png"
    plot_enrichment_heatmap(df, labels, out)
    assert out.exists() and out.stat().st_size > 0


# ── edge cases ────────────────────────────────────────────────────────────────

def test_single_family_still_works(tmp_path):
    df = pd.DataFrame({"prompt_family": ["FamA"] * 20})
    labels = np.array([0] * 10 + [1] * 10)
    out = tmp_path / "enrich.png"
    ct = plot_enrichment_heatmap(df, labels, out)
    assert ct.shape[0] == 1
    assert abs(ct.iloc[0].sum() - 1.0) < 1e-6
