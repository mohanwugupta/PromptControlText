"""tests/test_mining_reports.py — Milestone 6 report artifact tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from mining.clustering import run_text_clustering, top_terms_per_cluster
from mining.exemplars import build_exemplar_table
from mining.reports import (
    plot_cluster_sizes,
    plot_enrichment_heatmap,
    write_cluster_report,
    write_routing_sensitivity_report,
    write_taxonomy_memo,
)
from mining.routing_sensitivity import (
    compute_routing_sensitivity,
    build_routing_sensitive_report,
)


def _simple_df(n: int = 60) -> pd.DataFrame:
    texts = (
        [f"I cannot help with this request {i}." for i in range(n // 2)]
        + [f"Sure, here is how to do task {i}. Step 1: proceed." for i in range(n // 2)]
    )
    return pd.DataFrame({
        "model_output": texts,
        "item_id": [f"item_{i:03d}" for i in range(n)],
        "benchmark": (["XSTest"] * (n // 4) + ["HarmBench"] * (n // 4)) * 2,
        "prompt_family": (["Refuse-first"] * (n // 2) + ["Helpful-baseline"] * (n // 2)),
        "gold_label": ["safe"] * (n // 2) + ["unsafe"] * (n // 2),
    })


# ── figure 1: cluster size bar chart ─────────────────────────────────────────

def test_cluster_size_figure_written(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    out = tmp_path / "fig1.png"
    plot_cluster_sizes(r, out)
    assert out.exists() and out.stat().st_size > 0


# ── figure 3: enrichment heatmap ─────────────────────────────────────────────

def test_enrichment_figure_written(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    out = tmp_path / "fig3.png"
    plot_enrichment_heatmap(df, r.labels, out)
    assert out.exists() and out.stat().st_size > 0


# ── cluster report text ───────────────────────────────────────────────────────

def test_cluster_report_written(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    terms = top_terms_per_cluster(r)
    exemplars = build_exemplar_table(df, r, terms)
    out = tmp_path / "cluster_report.txt"
    write_cluster_report(exemplars, terms, r, out)
    assert out.exists() and out.stat().st_size > 0


def test_cluster_report_contains_required_sections(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    terms = top_terms_per_cluster(r)
    exemplars = build_exemplar_table(df, r, terms)
    out = tmp_path / "cr.txt"
    write_cluster_report(exemplars, terms, r, out)
    text = out.read_text()
    assert "CLUSTER REPORT" in text
    assert "Top terms" in text
    assert "PENDING MANUAL REVIEW" in text


# ── routing-sensitive item report ─────────────────────────────────────────────

def test_routing_sensitivity_report_written(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    sensitivity = compute_routing_sensitivity(df, r.labels)
    report_df = build_routing_sensitive_report(df, sensitivity, r.labels, top_n=5)
    out = tmp_path / "rs_report.txt"
    write_routing_sensitivity_report(report_df, sensitivity, out)
    assert out.exists() and out.stat().st_size > 0


def test_routing_sensitivity_report_has_required_columns(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    sensitivity = compute_routing_sensitivity(df, r.labels)
    report_df = build_routing_sensitive_report(df, sensitivity, r.labels, top_n=5)
    out = tmp_path / "rs_report.txt"
    write_routing_sensitivity_report(report_df, sensitivity, out)
    text = out.read_text()
    assert "disagreement" in text
    assert "item_id" in text


# ── taxonomy memo ─────────────────────────────────────────────────────────────

def test_taxonomy_memo_written(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    terms = top_terms_per_cluster(r)
    out = tmp_path / "taxonomy.txt"
    write_taxonomy_memo(terms, r, out)
    assert out.exists() and out.stat().st_size > 0


def test_taxonomy_memo_contains_v2_labels(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    terms = top_terms_per_cluster(r)
    out = tmp_path / "taxonomy.txt"
    write_taxonomy_memo(terms, r, out)
    text = out.read_text()
    for label in ["refuse", "clarify", "answer"]:
        assert label in text


def test_taxonomy_memo_fill_in_placeholders(tmp_path):
    df = _simple_df()
    r = run_text_clustering(df, n_clusters=2, random_state=42)
    terms = top_terms_per_cluster(r)
    out = tmp_path / "taxonomy.txt"
    write_taxonomy_memo(terms, r, out)
    text = out.read_text()
    assert "FILL IN" in text
