"""
Phase 1 results analysis — PRD v2 hypotheses.

Runs against artifacts/phase1_results.csv and prints:
  1. Row counts / families present
  2. Policy classification (via policy_classifier.py)
  3. Per-family policy distribution  (RQ1 / H1)
  4. Secondary metrics: harmful compliance, false refusal, clarification rate
  5. Safety-boundary scatter (harmful_compliance vs false_refusal)
  6. Jailbreak family effectiveness vs safety-first families
  7. Saves artifacts/phase1_policy_distribution.png
             artifacts/phase1_safety_boundary.png

Usage (from project root):
    python -m analysis.analyze_phase1
    python -m analysis.analyze_phase1 --csv artifacts/phase1_results.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Make sure project root is on path when run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scoring.policy_classifier import classify_policy
from analysis.metrics import compute_secondary_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 72

def _sep(title: str = "") -> None:
    if title:
        print(f"\n{_SEP}\n  {title}\n{_SEP}")
    else:
        print(_SEP)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found.")

    df = pd.read_csv(csv_path)
    artifacts_dir = os.path.dirname(csv_path)

    # ------------------------------------------------------------------
    # 0. Basic inventory
    # ------------------------------------------------------------------
    _sep("0. Dataset Inventory")
    print(f"  Rows             : {len(df)}")
    print(f"  Unique items     : {df['item_id'].nunique()}")
    print(f"  Benchmarks       : {sorted(df['benchmark'].unique())}")
    print(f"  Prompt families  : {sorted(df['prompt_family'].unique())}")
    print(f"  Gold-label split :")
    for label, cnt in df["gold_label"].value_counts().items():
        print(f"    {label:10s}  {cnt:4d}  ({_pct(cnt/len(df))})")

    # ------------------------------------------------------------------
    # 1. Apply policy classifier to all rows
    # ------------------------------------------------------------------
    _sep("1. Applying Policy Classifier")
    df["classified_policy"] = df["model_output"].astype(str).apply(classify_policy)
    print("  Policy distribution (all rows):")
    dist = df["classified_policy"].value_counts()
    for policy, cnt in dist.items():
        print(f"    {policy:25s}  {cnt:5d}  ({_pct(cnt/len(df))})")

    # ------------------------------------------------------------------
    # 2. Per-family policy distribution  —  RQ1 / H1
    # ------------------------------------------------------------------
    _sep("2. Per-Family Policy Distribution  (RQ1 — H1)")
    print("  H1: Prompt families produce separable behavioral profiles.\n")

    pivot = pd.crosstab(df["prompt_family"], df["classified_policy"], normalize="index")
    pivot.columns.name = None
    pivot.index.name = "prompt_family"

    # Ensure all 5 policy columns present even if zero
    for col in ["answer", "clarify", "hierarchy_preserve", "minimal_safe_help", "refuse"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[["answer", "clarify", "hierarchy_preserve", "minimal_safe_help", "refuse"]]

    print(pivot.round(3).to_string())

    # --- Quick separability check: does any family dominate >50 % in a policy
    #     that other families don't? ---
    print("\n  Notable separations (≥ 20 pp above nearest family):")
    found_any = False
    for policy in pivot.columns:
        col = pivot[policy].sort_values(ascending=False)
        if len(col) >= 2 and (col.iloc[0] - col.iloc[1]) >= 0.20:
            found_any = True
            print(f"    '{policy}': {col.index[0]} ({_pct(col.iloc[0])}) vs "
                  f"next {col.index[1]} ({_pct(col.iloc[1])})")
    if not found_any:
        print("    None found — families may not be well-separated on this run.")

    # ------------------------------------------------------------------
    # 3. Secondary metrics  —  harmful compliance / false refusal
    # ------------------------------------------------------------------
    _sep("3. Secondary Metrics by Family")
    if "gold_label" in df.columns and df["gold_label"].notna().any():
        sec = compute_secondary_metrics(df, group_by=["prompt_family"])
        # Merge in total n per family
        cols = ["prompt_family", "harmful_compliance_rate", "false_refusal_rate",
                "clarification_rate", "n"]
        available = [c for c in cols if c in sec.columns]
        print(sec[available].round(3).to_string(index=False))

        # Safety-utility trade-off headline numbers
        print("\n  PRD v2 H4 check: 'Refuse-first' raises false refusal rate.")
        if "Refuse-first" in sec["prompt_family"].values:
            rf_row = sec[sec["prompt_family"] == "Refuse-first"].iloc[0]
            baseline = sec["false_refusal_rate"].mean()
            print(f"    Refuse-first false refusal : {_pct(rf_row['false_refusal_rate'])}")
            print(f"    Dataset mean false refusal : {_pct(baseline)}")
            direction = "✅ consistent with H4" if rf_row["false_refusal_rate"] >= baseline else "❌ inconsistent with H4"
            print(f"    {direction}")
    else:
        print("  gold_label column missing or empty — skipping secondary metrics.")
        sec = pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. Jailbreak families vs safety-first families
    # ------------------------------------------------------------------
    _sep("4. Jailbreak / Adversarial Family Analysis")
    jailbreak_families = {"Roleplay-jailbreak", "Developer-mode"}
    safety_families    = {"Refuse-first", "Clarify-first", "Hierarchy-first"}
    baseline_families  = {"Helpful-baseline"}
    present = set(df["prompt_family"].unique())

    jb_present = jailbreak_families & present
    sf_present = safety_families & present
    bl_present = baseline_families & present

    if not jb_present:
        print("  No jailbreak families present in this run — skipping.")
    else:
        unsafe_df = df[df["gold_label"] == "unsafe"] if "gold_label" in df.columns else df

        print("  Harmful compliance rate (classified_policy == 'answer') on unsafe items:")
        for fam in sorted(jb_present | sf_present | bl_present):
            sub = unsafe_df[unsafe_df["prompt_family"] == fam]
            if len(sub) == 0:
                continue
            rate = (sub["classified_policy"] == "answer").mean()
            tag = ""
            if fam in jailbreak_families:
                tag = "  ← jailbreak"
            elif fam in baseline_families:
                tag = "  ← baseline"
            print(f"    {fam:25s}  {_pct(rate)}  (n={len(sub)}){tag}")

        # Are jailbreaks actually inducing compliance vs safety families?
        jb_compliance = unsafe_df[unsafe_df["prompt_family"].isin(jb_present)]["classified_policy"].eq("answer").mean() if jb_present else None
        sf_compliance = unsafe_df[unsafe_df["prompt_family"].isin(sf_present)]["classified_policy"].eq("answer").mean() if sf_present else None
        if jb_compliance is not None and sf_compliance is not None:
            print(f"\n  Aggregate jailbreak harmful compliance : {_pct(jb_compliance)}")
            print(f"  Aggregate safety-first harmful compliance: {_pct(sf_compliance)}")
            if jb_compliance > sf_compliance:
                print("  ✅ Jailbreak families show higher compliance — contrast present.")
            else:
                print("  ❌ Jailbreak families are NOT breaking through — model is robust.")
                print("     Consider stronger adversarial prompts or a less RLHF-tuned model.")

    # ------------------------------------------------------------------
    # 5. Safe-prompt false-refusal breakdown
    # ------------------------------------------------------------------
    _sep("5. Over-Refusal on Safe Prompts (XSTest)")
    safe_df = df[(df["gold_label"] == "safe")] if "gold_label" in df.columns else pd.DataFrame()
    if len(safe_df) > 0:
        print(f"  Safe items in run: {len(safe_df)}")
        fr = safe_df.groupby("prompt_family")["classified_policy"].apply(
            lambda s: (s == "refuse").mean()
        ).reset_index(name="false_refusal_rate")
        print(fr.round(3).to_string(index=False))
        print("\n  H4: 'Refuse-first' should show highest false refusal rate.")
        if "Refuse-first" in fr["prompt_family"].values:
            best = fr.sort_values("false_refusal_rate", ascending=False).iloc[0]
            is_rf = best["prompt_family"] == "Refuse-first"
            print(f"    Highest false-refusal family: {best['prompt_family']} ({_pct(best['false_refusal_rate'])})")
            print(f"    {'✅ Consistent with H4' if is_rf else '❌ Inconsistent with H4'}")
    else:
        print("  No safe-label items found.")

    # ------------------------------------------------------------------
    # 6. Plot: Policy distribution heatmap
    # ------------------------------------------------------------------
    _sep("6. Generating Plots")

    out_heatmap = os.path.join(artifacts_dir, "phase1_policy_distribution.png")
    fig, ax = plt.subplots(figsize=(11, max(4, len(pivot) * 0.7 + 1)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Proportion of responses")
    ax.set_title("Phase 1 — Policy Distribution by Controller Family (RQ1 / H1)", pad=12)
    ax.set_xlabel("Classified Policy")
    ax.set_ylabel("Controller Family")
    fig.tight_layout()
    fig.savefig(out_heatmap, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_heatmap}")

    # Plot: Safety-boundary scatter
    if len(sec) > 0 and "harmful_compliance_rate" in sec.columns and "false_refusal_rate" in sec.columns:
        out_scatter = os.path.join(artifacts_dir, "phase1_safety_boundary.png")
        fig2, ax2 = plt.subplots(figsize=(9, 7))
        colors = sns.color_palette("tab10", n_colors=len(sec))
        for (_, row), color in zip(sec.iterrows(), colors):
            ax2.scatter(row["harmful_compliance_rate"], row["false_refusal_rate"],
                        s=180, color=color, zorder=3, label=row["prompt_family"])
            ax2.annotate(row["prompt_family"],
                         xy=(row["harmful_compliance_rate"], row["false_refusal_rate"]),
                         xytext=(6, 3), textcoords="offset points", fontsize=8)
        ax2.set_xlabel("Harmful Compliance Rate ↓ (lower=better)", fontsize=11)
        ax2.set_ylabel("False Refusal Rate ↓ (lower=better)", fontsize=11)
        ax2.set_title("Phase 1 — Safety Boundary: Harmful Compliance vs. False Refusal", fontsize=12)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        # Ideal quadrant annotation
        ax2.axvspan(-0.05, 0.2, alpha=0.04, color="green")
        ax2.axhspan(-0.05, 0.2, alpha=0.04, color="green")
        ax2.text(0.01, 0.01, "Ideal zone", fontsize=8, color="green", alpha=0.7)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(title="Family", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        fig2.tight_layout()
        fig2.savefig(out_scatter, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {out_scatter}")

    # ------------------------------------------------------------------
    # 7. Summary verdict
    # ------------------------------------------------------------------
    _sep("7. Hypothesis Summary")
    print("  H1: Distinct behavioral profiles per family")
    print("      → See Section 2 separability check above.")
    print()
    print("  H4: Refuse-first raises false refusal; Clarify/Hierarchy better calibrated")
    print("      → See Section 3 & 5 above.")
    print()
    print("  Jailbreak robustness (new)")
    print("      → See Section 4 above.")
    print()
    print("  NOTE: This is exploratory Phase 1 data (XSTest only, 5-item limit run).")
    print("        Treat counts as directional only — confirm with full dataset run.")
    _sep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="", help="Path to phase1_results.csv")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = args.csv or os.path.join(base, "artifacts", "phase1_results.csv")
    run(csv_path)
