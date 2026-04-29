"""
mining/reports.py
=================
Milestone 6 — Report and figure generation (PRD v3 §12, §16).

Produces:
  1. Cluster report CSV + text summary
  2. Cluster enrichment heatmap (Figure 3)
  3. Cluster size distribution bar chart (Figure 1)
  4. 2-D scatter of outputs coloured by cluster (Figure 2)
  5. Routing-sensitive item summary CSV (Figure 4)
  6. Taxonomy memo scaffold (text file)

All outputs are saved with ISO-date-stamped filenames under
``artifacts/mining/<run_date>/``.

CLI usage::

    python -m mining.reports --phase1 artifacts/phase1_results.csv \\
                             --phase2 artifacts/phase2_results.csv \\
                             --n-clusters 7 \\
                             --out-dir artifacts/mining

Programmatic usage::

    from mining.reports import run_full_pipeline
    run_full_pipeline(phase1_path, phase2_path, out_dir, n_clusters=7)
"""

from __future__ import annotations

import argparse
import datetime
import json
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mining.response_table import build_response_table
from mining.features import extract_features_df, FEATURE_NUMERIC_COLS
from mining.clustering import (
    run_text_clustering,
    run_hybrid_clustering,
    top_terms_per_cluster,
    ClusterResult,
)
from mining.exemplars import build_exemplar_table
from mining.routing_sensitivity import (
    compute_routing_sensitivity,
    get_routing_sensitive_items,
    build_routing_sensitive_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dated_dir(out_dir: Path) -> Path:
    stamp = datetime.date.today().isoformat()
    d = out_dir / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


# ---------------------------------------------------------------------------
# Figure 1: Cluster size distribution
# ---------------------------------------------------------------------------

def plot_cluster_sizes(result: ClusterResult, out_path: Path) -> None:
    sizes = result.cluster_sizes
    labels = [f"C{k}" for k in sorted(sizes)]
    counts = [sizes[k] for k in sorted(sizes)]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
    bars = ax.bar(labels, counts, color=sns.color_palette("tab10", len(labels)))
    ax.bar_label(bars)
    ax.set_title(f"Cluster size distribution  (mode={result.mode}, k={result.n_clusters})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("# responses")
    ax.set_ylim(0, max(counts) * 1.15)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 2: 2-D scatter
# ---------------------------------------------------------------------------

def plot_2d_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str = "2-D projection coloured by cluster",
) -> None:
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("tab10", n_clusters)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in range(n_clusters):
        mask = labels[:len(coords)] == cid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, alpha=0.5, color=palette[cid], label=f"C{cid}",
        )
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 3: Enrichment heatmap
# ---------------------------------------------------------------------------

def plot_enrichment_heatmap(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    out_path: Path,
    family_col: str = "prompt_family",
) -> pd.DataFrame:
    """
    Normalised cluster enrichment by prompt family.
    Returns the contingency DataFrame (rows=family, cols=cluster).
    """
    work = df[[family_col]].copy().reset_index(drop=True)
    work["cluster"] = cluster_labels
    ct = pd.crosstab(work[family_col], work["cluster"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)  # row-normalise

    fig, ax = plt.subplots(figsize=(max(6, ct_norm.shape[1] * 0.9),
                                    max(4, ct_norm.shape[0] * 0.5)))
    sns.heatmap(
        ct_norm, annot=True, fmt=".2f", cmap="Blues",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "proportion of family outputs"},
    )
    ax.set_title("Cluster enrichment by prompt family (row-normalised)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Prompt family")
    plt.xticks(
        ticks=np.arange(ct_norm.shape[1]) + 0.5,
        labels=[f"C{c}" for c in ct_norm.columns],
        rotation=0,
    )
    _save(fig, out_path)
    return ct_norm


# ---------------------------------------------------------------------------
# Cluster report (Output 1)
# ---------------------------------------------------------------------------

def write_cluster_report(
    exemplar_table: pd.DataFrame,
    top_terms: dict,
    result: ClusterResult,
    out_path: Path,
) -> None:
    lines = [
        "=" * 72,
        f"CLUSTER REPORT  (mode={result.mode}, k={result.n_clusters}, "
        f"silhouette={result.silhouette:.3f})",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        "",
    ]
    for cid in range(result.n_clusters):
        size = result.cluster_sizes.get(cid, 0)
        terms = ", ".join(top_terms.get(cid, []))
        lines += [
            f"── Cluster {cid}  (n={size}) ──",
            f"   Top terms : {terms}",
            f"   Suggested label: [PENDING MANUAL REVIEW]",
            "",
        ]
        exemplars = exemplar_table[
            (exemplar_table["cluster"] == cid)
            & (exemplar_table["selection_method"] == "centroid")
        ]
        for _, row in exemplars.iterrows():
            txt = str(row.get("model_output", ""))[:300]
            fam = row.get("prompt_family", "")
            bench = row.get("benchmark", "")
            lines += [
                f"   [{bench} | {fam}]",
                textwrap.fill(txt, width=68, initial_indent="   > ",
                              subsequent_indent="     "),
                "",
            ]
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Routing-sensitive item report (Output 2)
# ---------------------------------------------------------------------------

def write_routing_sensitivity_report(
    report_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "=" * 72,
        "ROUTING-SENSITIVE ITEM REPORT",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        f"Top {len(sensitivity_df.head(20))} items by disagreement_score",
        "",
    ]
    for _, row in sensitivity_df.head(20).iterrows():
        lines += [
            f"  item_id          : {row['item_id']}",
            f"  disagreement     : {row['disagreement_score']:.3f}",
            f"  n_families       : {row['n_families']}",
            f"  cluster_entropy  : {row['cluster_entropy']:.3f}",
            f"  policy_entropy   : {row.get('policy_entropy', 'n/a')}",
            "",
        ]
        item_outputs = report_df[report_df["item_id"] == row["item_id"]]
        for _, orow in item_outputs.iterrows():
            fam = orow.get("prompt_family", "")
            cid = orow.get("cluster", "?")
            txt = str(orow.get("model_output", ""))[:200]
            lines += [
                f"    [{fam}  → C{cid}]",
                textwrap.fill(txt, width=68, initial_indent="    > ",
                              subsequent_indent="      "),
                "",
            ]
        lines.append("-" * 40)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Taxonomy memo scaffold (Output 3)
# ---------------------------------------------------------------------------

def write_taxonomy_memo(
    top_terms: dict,
    result: ClusterResult,
    out_path: Path,
) -> None:
    CANDIDATE_LABELS = [
        "direct refusal",
        "hedged refusal",
        "clarification",
        "safe redirection",
        "minimal-safe-help",
        "hierarchy-preserving refusal",
        "answer with safety preamble",
        "direct compliance",
        "mixed / ambiguous",
    ]

    v2_labels = ["answer", "refuse", "clarify", "minimal_safe_help", "hierarchy_preserve"]

    lines = [
        "=" * 72,
        "TAXONOMY REVISION MEMO  [DRAFT — pending manual exemplar review]",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        "",
        "Current v2 policy labels:",
        "  " + ", ".join(v2_labels),
        "",
        "Cluster → provisional label mapping",
        "(Fill in after manual exemplar inspection)",
        "-" * 40,
    ]
    for cid in range(result.n_clusters):
        size = result.cluster_sizes.get(cid, 0)
        terms = ", ".join(top_terms.get(cid, [])[:8])
        lines += [
            f"  Cluster {cid}  (n={size})",
            f"    Top terms    : {terms}",
            f"    Provisional label : [FILL IN]",
            f"    Maps to v2   : [FILL IN or 'NEW']",
            f"    Notes        : ",
            "",
        ]

    lines += [
        "-" * 40,
        "",
        "Candidate labels (from PRD v3 §11):",
        "  " + "\n  ".join(CANDIDATE_LABELS),
        "",
        "Assessment questions:",
        "  1. Are all v2 labels represented? Which are missing?",
        "  2. Are any clusters not covered by v2 labels?",
        "  3. Should any v2 labels be merged or split?",
        "  4. Are any clusters too heterogeneous to label?",
        "",
        "[END OF MEMO]",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    phase1_path: str,
    phase2_path: str,
    out_dir: str = "artifacts/mining",
    *,
    n_clusters: int = 7,
    random_state: int = 42,
    run_viz: bool = False,
    top_n_sensitive: int = 50,
) -> dict:
    """
    Execute the complete mining pipeline and write all outputs.

    Parameters
    ----------
    phase1_path, phase2_path : str
        Paths to results CSVs.
    out_dir : str
        Base output directory (subdirectory per date is created).
    n_clusters : int
        Number of clusters for KMeans.
    run_viz : bool
        If True, also generate 2-D scatter (slow for large corpora).

    Returns
    -------
    dict with paths to every written artifact.
    """
    print(f"\n{'='*60}")
    print(f"PRD v3 Mining Pipeline   n_clusters={n_clusters}")
    print(f"{'='*60}\n")

    out = _dated_dir(Path(out_dir))
    artifacts: dict = {}

    # ── Step 1: build unified table ──────────────────────────────────────
    print("[1/7] Building unified response table …")
    df = build_response_table(phase1_path, phase2_path)
    print(f"      {len(df):,} rows  ({df['phase'].value_counts().to_dict()})")

    # ── Step 2: feature extraction ───────────────────────────────────────
    print("[2/7] Extracting text features …")
    df = extract_features_df(df)

    # ── Step 3: text-only clustering ─────────────────────────────────────
    print("[3/7] Text-only clustering …")
    text_result = run_text_clustering(df, n_clusters, random_state=random_state)
    df["cluster_text"] = text_result.labels
    print(f"      silhouette={text_result.silhouette:.3f}  sizes={text_result.cluster_sizes}")

    # ── Step 4: hybrid clustering ─────────────────────────────────────────
    print("[4/7] Hybrid clustering …")
    hybrid_result = run_hybrid_clustering(df, n_clusters, random_state=random_state)
    df["cluster_hybrid"] = hybrid_result.labels
    print(f"      silhouette={hybrid_result.silhouette:.3f}  sizes={hybrid_result.cluster_sizes}")

    # Save enriched table
    table_path = out / "mining_table.csv"
    df.to_csv(table_path, index=False)
    artifacts["mining_table"] = str(table_path)
    print(f"      table saved → {table_path.name}")

    # ── Step 5: exemplars ─────────────────────────────────────────────────
    print("[5/7] Selecting exemplars …")
    top_terms_text = top_terms_per_cluster(text_result)
    exemplar_tbl = build_exemplar_table(df, text_result, top_terms_text,
                                        random_state=random_state)
    ex_path = out / "exemplars_text_clustering.csv"
    exemplar_tbl.to_csv(ex_path, index=False)
    artifacts["exemplars"] = str(ex_path)

    # ── Step 6: routing sensitivity ───────────────────────────────────────
    print("[6/7] Computing routing sensitivity …")
    sensitivity = compute_routing_sensitivity(df, text_result.labels,
                                              char_col="char_count")
    top_sensitive = get_routing_sensitive_items(sensitivity, top_n=top_n_sensitive)
    sens_path = out / "routing_sensitivity.csv"
    sensitivity.to_csv(sens_path, index=False)
    artifacts["sensitivity"] = str(sens_path)

    rpt_df = build_routing_sensitive_report(
        df, sensitivity, text_result.labels, top_n=20
    )

    # ── Step 7: reports and figures ───────────────────────────────────────
    print("[7/7] Generating reports and figures …")

    # Figure 1 – cluster sizes
    fig1_path = out / "fig1_cluster_sizes.png"
    plot_cluster_sizes(text_result, fig1_path)
    artifacts["fig_cluster_sizes"] = str(fig1_path)

    # Figure 2 – 2D scatter (optional, expensive)
    if run_viz:
        from mining.clustering import reduce_for_viz
        print("      reducing to 2D (t-SNE) …")
        coords = reduce_for_viz(text_result, method="tsne", random_state=random_state)
        fig2_path = out / "fig2_2d_scatter.png"
        plot_2d_scatter(coords, text_result.labels, fig2_path)
        artifacts["fig_2d_scatter"] = str(fig2_path)

    # Figure 3 – enrichment heatmap
    fig3_path = out / "fig3_enrichment_heatmap.png"
    enrichment = plot_enrichment_heatmap(df, text_result.labels, fig3_path)
    enrich_path = out / "enrichment_table.csv"
    enrichment.to_csv(enrich_path)
    artifacts["enrichment_table"] = str(enrich_path)
    artifacts["fig_enrichment"] = str(fig3_path)

    # Cluster report text
    cr_path = out / "cluster_report.txt"
    write_cluster_report(exemplar_tbl, top_terms_text, text_result, cr_path)
    artifacts["cluster_report"] = str(cr_path)

    # Routing-sensitive report text
    rs_path = out / "routing_sensitive_report.txt"
    write_routing_sensitivity_report(rpt_df, top_sensitive, rs_path)
    artifacts["routing_sensitive_report"] = str(rs_path)

    # Taxonomy memo
    tm_path = out / "taxonomy_memo.txt"
    write_taxonomy_memo(top_terms_text, text_result, tm_path)
    artifacts["taxonomy_memo"] = str(tm_path)

    # Manifest
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    print(f"\n✓ All outputs in: {out}\n")
    return artifacts


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRD v3 Mining Pipeline")
    parser.add_argument("--phase1", default="artifacts/phase1_results.csv")
    parser.add_argument("--phase2", default="artifacts/phase2_results.csv")
    parser.add_argument("--out-dir", default="artifacts/mining")
    parser.add_argument("--n-clusters", type=int, default=7)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--viz", action="store_true",
                        help="Also generate 2D scatter (slow)")
    args = parser.parse_args()

    run_full_pipeline(
        args.phase1,
        args.phase2,
        args.out_dir,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        run_viz=args.viz,
    )
