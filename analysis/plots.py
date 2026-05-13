"""
analysis/plots.py
=================
Publication-quality figures for the policy-routing experiment.

All figures read from the validated, re-scored mining table via
``analysis.metrics.load_mining_table()``.  Each function saves a 300-dpi
PNG and returns the (fig, ax) pair for programmatic access.

Figure catalogue
----------------
plot_policy_routing_stacked      Family × policy label (stacked bar, 95% CI whiskers)
plot_safety_boundary             Harmful-compliance vs false-refusal scatter
plot_hierarchy_conflict          IHEval hierarchy-following rates by family
plot_refusal_by_benchmark        Refusal rate family × benchmark (grouped bar)
plot_routing_entropy_hist        Per-item routing entropy histogram
plot_cohens_h_heatmap            Pairwise Cohen's h heatmap for each policy label
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import argparse
import pandas as pd

from analysis.metrics import load_mining_table
from analysis.statistical_model import (
    routing_effect_table, compute_pairwise_cohens_h,
    compute_family_effects_summary_from_df,
)

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

FAMILY_ORDER = [
    "Refuse-first", "Hierarchy-first", "Helpful-baseline",
    "Developer-mode", "Clarify-first", "Roleplay-jailbreak",
]

# One colour per family — consistent across all figures
FAMILY_COLORS = {
    "Refuse-first":      "#c0392b",
    "Hierarchy-first":   "#8e44ad",
    "Helpful-baseline":  "#2980b9",
    "Developer-mode":    "#16a085",
    "Clarify-first":     "#f39c12",
    "Roleplay-jailbreak":"#7f8c8d",
}

# Policy-label colours for stacked bars
POLICY_COLORS = {
    "direct_answer":      "#2ecc71",
    "clarification":      "#3498db",
    "refusal":            "#e74c3c",
    "safe_partial":       "#f39c12",
    "hierarchy_defer":    "#8e44ad",
    "classified_refusal": "#16a085",
    "mixed_or_unclear":   "#95a5a6",
}

POLICY_DISPLAY = {
    "direct_answer":      "Direct answer",
    "clarification":      "Clarification",
    "refusal":            "Refusal",
    "safe_partial":       "Safe/partial",
    "hierarchy_defer":    "Hierarchy defer",
    "classified_refusal": "Classified refusal",
    "mixed_or_unclear":   "Mixed/unclear",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _add_ci_whiskers(ax, x_positions, proportions, ci_lowers, ci_uppers, bottoms=None):
    """Draw symmetric CI whiskers on a stacked bar chart."""
    for x, p, lo, hi, b in zip(
        x_positions, proportions, ci_lowers, ci_uppers,
        bottoms if bottoms is not None else [0] * len(x_positions),
    ):
        mid = b + p
        ax.errorbar(x, mid, yerr=[[mid - (b + lo)], [(b + hi) - mid]],
                    fmt="none", color="black", capsize=3, linewidth=0.8, alpha=0.6)


# ---------------------------------------------------------------------------
# Figure 1: Policy routing stacked bar (main result figure)
# ---------------------------------------------------------------------------

def plot_policy_routing_stacked(
    output_path: str = "artifacts/audit/figures/fig_policy_routing.png",
    mining_table: str = None,
    benchmark: str = None,
) -> tuple:
    """Stacked bar chart of policy-label proportions by prompt family,
    with 95% Wilson CI whiskers on each segment.

    Parameters
    ----------
    benchmark : filter to one benchmark (None = all).
    """
    df = load_mining_table(mining_table) if mining_table else load_mining_table()
    if benchmark:
        df = df[df["benchmark"] == benchmark]
    else:
        # IHEval uses a completely different label taxonomy (hierarchy task).
        # Exclude it from the main safety-routing chart; it appears in
        # plot_hierarchy_conflict instead.
        df = df[df["benchmark"] != "IHEval"]

    tbl = routing_effect_table(df, family_order=FAMILY_ORDER)
    labels_present = sorted(df["primary_policy_label"].unique())
    labels_ordered = [l for l in POLICY_COLORS if l in labels_present]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(FAMILY_ORDER))
    bottoms = np.zeros(len(FAMILY_ORDER))

    for lbl in labels_ordered:
        vals, ci_lo, ci_hi = [], [], []
        for fam in FAMILY_ORDER:
            try:
                row = tbl.loc[(fam, lbl)]
                vals.append(row["proportion"])
                ci_lo.append(row["ci_lower"])
                ci_hi.append(row["ci_upper"])
            except KeyError:
                vals.append(0.0); ci_lo.append(0.0); ci_hi.append(0.0)
        vals = np.array(vals)
        color = POLICY_COLORS.get(lbl, "#bdc3c7")
        ax.bar(x, vals, bottom=bottoms, color=color,
               label=POLICY_DISPLAY.get(lbl, lbl), width=0.6)
        # Label inside bar if segment > 5%
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.05:
                ax.text(i, b + v / 2, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        _add_ci_whiskers(ax, x, vals, ci_lo, ci_hi, bottoms)
        bottoms += vals

    bm_label = f" — {benchmark}" if benchmark else " — HarmBench + XSTest"
    n_total = len(df)
    ax.set_title(
        f"Policy routing by prompt family{bm_label}  "
        f"(n={n_total:,}, scorer κ=0.865 vs human labels)"
    )
    ax.set_ylabel("Proportion of responses")
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_ORDER, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8,
              bbox_to_anchor=(1.30, 1), framealpha=0.9)
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2: Safety boundary scatter
# ---------------------------------------------------------------------------

def plot_safety_boundary(
    output_path: str = "artifacts/audit/figures/fig_safety_boundary.png",
    mining_table: str = None,
) -> tuple:
    """Harmful-compliance rate vs false-refusal rate per prompt family.

    Uses re-scored policy labels:
      harmful compliance = primary_policy_label == 'direct_answer'
                           on gold_label == 'unsafe' (HarmBench)
      false refusal      = primary_policy_label == 'refusal'
                           on gold_label == 'safe' (XSTest)
    """
    df = load_mining_table(mining_table) if mining_table else load_mining_table()

    unsafe = df[(df["benchmark"] == "HarmBench") & (df.get("gold_label", pd.Series()) == "unsafe")
                ] if "gold_label" in df.columns else df[df["benchmark"] == "HarmBench"]
    safe   = df[(df["benchmark"] == "XSTest")   & (df.get("gold_label", pd.Series()) == "safe")
                ] if "gold_label" in df.columns else df[df["benchmark"] == "XSTest"]

    hc = unsafe.groupby("prompt_family").apply(
        lambda g: (g["primary_policy_label"] == "direct_answer").mean(),
        include_groups=False,
    ).rename("harmful_compliance")
    fr = safe.groupby("prompt_family").apply(
        lambda g: (g["primary_policy_label"] == "refusal").mean(),
        include_groups=False,
    ).rename("false_refusal")
    merged = pd.concat([hc, fr], axis=1).fillna(0).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in merged.iterrows():
        fam = row["prompt_family"]
        color = FAMILY_COLORS.get(fam, "#95a5a6")
        ax.scatter(row["harmful_compliance"], row["false_refusal"],
                   s=180, color=color, zorder=3, label=fam)
        ax.annotate(fam, (row["harmful_compliance"], row["false_refusal"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Harmful compliance rate  ← lower is better")
    ax.set_ylabel("False refusal rate  ← lower is better")
    ax.set_title("Safety boundary: harmful compliance vs false refusal by family")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 3: IHEval hierarchy conflict
# ---------------------------------------------------------------------------

def plot_hierarchy_conflict(
    output_path: str = "artifacts/audit/figures/fig_hierarchy_conflict.png",
    mining_table: str = None,
) -> tuple:
    """Hierarchy-following vs unsafe-continuation rates on IHEval by family."""
    df = load_mining_table(mining_table) if mining_table else load_mining_table()
    iheval = df[df["benchmark"] == "IHEval"].copy()

    score_cols = ["hierarchy_following_score", "unsafe_continuation_score", "stop_compliance_score"]
    score_cols = [c for c in score_cols if c in iheval.columns]
    agg = iheval.groupby("prompt_family")[score_cols].mean().reindex(FAMILY_ORDER).fillna(0)

    display_map = {
        "hierarchy_following_score": "Hierarchy following",
        "unsafe_continuation_score": "Unsafe continuation",
        "stop_compliance_score":     "Stop compliance",
    }
    colors_map = ["#2980b9", "#e74c3c", "#8e44ad"]

    x = np.arange(len(FAMILY_ORDER))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (col, color) in enumerate(zip(score_cols, colors_map)):
        ax.bar(x + (i - len(score_cols)/2 + 0.5) * w,
               agg[col].values, w,
               label=display_map.get(col, col), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.05)
    ax.set_title("IHEval hierarchy-conflict scores by prompt family")
    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 4: Refusal rate grouped by benchmark
# ---------------------------------------------------------------------------

def plot_refusal_by_benchmark(
    output_path: str = "artifacts/audit/figures/fig_refusal_by_benchmark.png",
    mining_table: str = None,
) -> tuple:
    """Grouped bar: refusal rate per family, one bar-cluster per benchmark."""
    df = load_mining_table(mining_table) if mining_table else load_mining_table()
    benchmarks = [b for b in ["HarmBench", "XSTest", "IHEval"] if b in df["benchmark"].unique()]
    bm_colors = {"HarmBench": "#e05c5c", "XSTest": "#5c8ce0", "IHEval": "#5ca05c"}

    x = np.arange(len(FAMILY_ORDER))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, bm in enumerate(benchmarks):
        sub = df[df["benchmark"] == bm]
        rates = sub.groupby("prompt_family").apply(
            lambda g: (g["primary_policy_label"] == "refusal").mean(),
            include_groups=False,
        ).reindex(FAMILY_ORDER).fillna(0)
        ax.bar(x + (i - len(benchmarks)/2 + 0.5) * w,
               rates.values, w, label=bm, color=bm_colors.get(bm, "#aaa"))

    ax.set_xticks(x)
    ax.set_xticklabels(FAMILY_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Refusal rate")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_title("Refusal rate by prompt family × benchmark  (scorer κ=0.865)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 5: Per-item routing entropy histogram
# ---------------------------------------------------------------------------

def plot_routing_entropy_hist(
    output_path: str = "artifacts/audit/figures/fig_routing_entropy.png",
    mining_table: str = None,
) -> tuple:
    """Histogram of per-item policy entropy across the 6 families."""
    from audit.analysis import compute_policy_entropy

    df = load_mining_table(mining_table) if mining_table else load_mining_table()
    entropy = compute_policy_entropy(df)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(entropy, bins=40, color="#7b5ea7", edgecolor="white", linewidth=0.4)
    mean_e = entropy.mean()
    max_e = np.log2(df["primary_policy_label"].nunique())
    ax.axvline(mean_e, color="#e74c3c", linewidth=1.5, linestyle="--",
               label=f"Mean = {mean_e:.3f} bits")
    ax.set_xlabel("Per-item routing entropy (bits)")
    ax.set_ylabel("Number of items")
    ax.set_title(
        f"Per-item policy routing entropy  "
        f"(max possible = {max_e:.2f} bits for "
        f"{df['primary_policy_label'].nunique()} labels)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 6: Cohen's h heatmap
# ---------------------------------------------------------------------------

def plot_cohens_h_heatmap(
    output_path: str = "artifacts/audit/figures/fig_cohens_h.png",
    mining_table: str = None,
    label: str = "refusal",
) -> tuple:
    """Heatmap of pairwise Cohen's h between families for a given policy label."""
    from audit.analysis import compute_family_effects_summary

    df = load_mining_table(mining_table) if mining_table else load_mining_table()
    fe = compute_family_effects_summary(df).reindex(FAMILY_ORDER)

    if label not in fe.columns:
        raise ValueError(f"Label '{label}' not in family-effects table. "
                         f"Available: {fe.columns.tolist()}")

    # Build pairwise matrix for this label
    families = FAMILY_ORDER
    mat = np.zeros((len(families), len(families)))
    for i, fa in enumerate(families):
        for j, fb in enumerate(families):
            mat[i, j] = abs(
                2 * np.arcsin(np.sqrt(fe.loc[fa, label])) -
                2 * np.arcsin(np.sqrt(fe.loc[fb, label]))
            )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1.2)
    ax.set_xticks(range(len(families))); ax.set_xticklabels(families, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(families))); ax.set_yticklabels(families, fontsize=8)
    for i in range(len(families)):
        for j in range(len(families)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="|Cohen's h|")
    ax.set_title(
        f"Pairwise |Cohen's h| for '{POLICY_DISPLAY.get(label, label)}'\n"
        f"(small≥0.20, medium≥0.50, large≥0.80)"
    )
    fig.tight_layout()
    _ensure_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return fig, ax


# ---------------------------------------------------------------------------
# Legacy entry point (kept for backward compatibility with SLURM scripts)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", type=str, default="all",
                        choices=["all", "routing", "safety", "hierarchy",
                                 "refusal", "entropy", "cohens_h"])
    parser.add_argument("--output-dir", type=str,
                        default="artifacts/audit/figures")
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()

    out = args.output_dir
    figs = {
        "routing":   lambda: plot_policy_routing_stacked(
            f"{out}/fig_policy_routing.png", benchmark=args.benchmark),
        "safety":    lambda: plot_safety_boundary(f"{out}/fig_safety_boundary.png"),
        "hierarchy": lambda: plot_hierarchy_conflict(f"{out}/fig_hierarchy_conflict.png"),
        "refusal":   lambda: plot_refusal_by_benchmark(f"{out}/fig_refusal_by_benchmark.png"),
        "entropy":   lambda: plot_routing_entropy_hist(f"{out}/fig_routing_entropy.png"),
        "cohens_h":  lambda: plot_cohens_h_heatmap(f"{out}/fig_cohens_h_refusal.png", label="refusal"),
    }

    to_run = figs.keys() if args.figure == "all" else [args.figure]
    for name in to_run:
        figs[name]()


def plot_safety_boundary(csv_path: str, output_path: str):
    """
    Reads the output from Phase 1 and plots Harmful Compliance vs False Refusal
    across prompt families. 
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Harmful compliance = Compliance on HarmBench 
    # False Refusal = Refusal on XSTest (safe prompts)
    
    # Calculate means per prompt family
    # Filter subsets
    unsafe_items = df[df["gold_label"] == "unsafe"]
    safe_items = df[df["gold_label"] == "safe"]
    
    harmful_compliance = unsafe_items.groupby("prompt_family")["compliance_score"].mean().reset_index()
    harmful_compliance.rename(columns={"compliance_score": "harmful_compliance_rate"}, inplace=True)
    
    false_refusals = safe_items.groupby("prompt_family")["refusal_score"].mean().reset_index()
    false_refusals.rename(columns={"refusal_score": "false_refusal_rate"}, inplace=True)
    
    merged = pd.merge(harmful_compliance, false_refusals, on="prompt_family", how="outer").fillna(0)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=merged, 
        x="harmful_compliance_rate", 
        y="false_refusal_rate", 
        hue="prompt_family", 
        s=200, 
        marker='o'
    )
    
    # Add labels
    for i in range(merged.shape[0]):
        plt.text(
            merged["harmful_compliance_rate"][i] + 0.02, 
            merged["false_refusal_rate"][i], 
            merged["prompt_family"][i], 
            horizontalalignment='left'
        )
        
    plt.title("Safety Boundary: Harmful Compliance vs. False Refusal")
    plt.xlabel("Harmful Compliance Rate (lower is better)")
    plt.ylabel("False Refusal Rate (lower is better)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Safety boundary plot saved to {output_path}")

def plot_hierarchy_conflict(csv_path: str, output_path: str):
    """
    Reads Phase 2 results and plots Accuracy / unsafe continuation rates 
    under vague vs explicit hierarchy prompts evaluating Conflict scenarios.
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter for conflicts
    conflict_df = df[df["gold_label"] == "conflict"]
    
    agg = conflict_df.groupby("prompt_family")[["hierarchy_following_score", "unsafe_continuation_score"]].mean().reset_index()
    agg.rename(columns={
        "hierarchy_following_score": "Accuracy (Hierarchy Following)",
        "unsafe_continuation_score": "Unsafe Continuation Rate"
    }, inplace=True)
    
    # Melt for grouped bar chart
    melted = agg.melt(id_vars="prompt_family", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="prompt_family", y="Score", hue="Metric")
    
    plt.title("Hierarchy Conflict Performance (IHEval Substrate)")
    plt.xlabel("Prompt Family")
    plt.ylabel("Average Score")
    plt.ylim(0, 1.1)
    plt.legend(title="Metric")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hierarchy conflict plot saved to {output_path}")


# ---------------------------------------------------------------------------
# PRD v2 plots
# ---------------------------------------------------------------------------

def plot_policy_distribution(csv_path: str, output_path: str):
    """Heatmap of policy-label distribution per controller family.

    PRD v2 §5 RQ1: Do prompt-level controllers induce distinct policy
    selections on the same item?
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Cross-tab: rows = prompt_family, cols = classified_policy
    ct = pd.crosstab(df["prompt_family"], df["classified_policy"], normalize="index")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)

    # Annotate cells
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)):
            val = ct.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=9)

    fig.colorbar(im, ax=ax, label="Proportion")
    ax.set_title("Policy Distribution by Controller Family (PRD v2 RQ1)")
    ax.set_xlabel("Classified Policy")
    ax.set_ylabel("Controller Family")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Policy distribution plot saved to {output_path}")


def plot_routing_accuracy(csv_path: str, output_path: str):
    """Grouped bar chart of routing accuracy by family × clarity level.

    PRD v2 §5 RQ2: Does controller clarity improve routing under
    ambiguity and conflict?
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    agg = df.groupby(["prompt_family", "clarity_level"])["routing_correct"].mean().reset_index()
    agg.rename(columns={"routing_correct": "Routing Accuracy"}, inplace=True)

    pivot = agg.pivot(index="prompt_family", columns="clarity_level", values="Routing Accuracy").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Routing Accuracy by Family × Clarity (PRD v2 RQ2)")
    ax.set_xlabel("Controller Family")
    ax.set_ylabel("Routing Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Clarity Level")
    plt.xticks(rotation=45, ha="right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Routing accuracy plot saved to {output_path}")


