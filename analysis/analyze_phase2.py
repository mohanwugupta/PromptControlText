"""
Phase 2 results analysis — PRD v2 Study 2 / RQ3 hypotheses.

Phase 2 tests hierarchy-sensitive control using IHEval. The benchmark
provides two item types:
  - conflict   : a lower-priority instruction attempts to override the
                 system prompt (e.g., "ignore previous instructions")
  - non-conflict: aligned instructions with no override attempt

The three scorer columns produced by hierarchy_scorer.py are:
  - hierarchy_following_score  : 1.0 = model correctly resisted override
  - unsafe_continuation_score  : 1.0 = model capitulated to the override
  - stop_compliance_score      : 1.0 = model stopped the unsafe action

Key hypotheses tested here:
  H3 (PRD v2): Hierarchy-first prompting outperforms refuse-first and
               vague safety prompts on conflict items specifically.
               The gap should be smaller on non-conflict items.
  H4 (PRD v2): Explicit controller wording improves routing under
               conflict more than on clean/non-conflict items.

Usage (from project root):
    python -m analysis.analyze_phase2
    python -m analysis.analyze_phase2 --csv artifacts/phase2_results.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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


def _delta(a: float, b: float) -> str:
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.1f} pp"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found.")

    df = pd.read_csv(csv_path)
    artifacts_dir = os.path.dirname(csv_path)

    SCORE_COLS = ["hierarchy_following_score", "unsafe_continuation_score", "stop_compliance_score"]
    for col in SCORE_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    # ------------------------------------------------------------------
    # 0. Basic inventory
    # ------------------------------------------------------------------
    _sep("0. Dataset Inventory")
    print(f"  Rows              : {len(df)}")
    print(f"  Unique items      : {df['item_id'].nunique()}")
    print(f"  Prompt families   : {sorted(df['prompt_family'].unique())}")
    print(f"  Prompt variants   : {sorted(df['prompt_variant'].unique())}")
    print(f"\n  Gold-label (conflict type) distribution:")
    for label, cnt in df["gold_label"].value_counts().items():
        print(f"    {label:15s}  {cnt:6d}  ({_pct(cnt / len(df))})")

    conflict_df     = df[df["gold_label"] == "conflict"]
    non_conflict_df = df[df["gold_label"] == "non-conflict"]
    print(f"\n  Conflict items    : {len(conflict_df)}")
    print(f"  Non-conflict items: {len(non_conflict_df)}")

    # ------------------------------------------------------------------
    # 1. Overall score distributions
    # ------------------------------------------------------------------
    _sep("1. Overall Score Distributions")
    for col in SCORE_COLS:
        if df[col].notna().any():
            dist = df[col].value_counts(normalize=True).sort_index()
            vals = "  ".join(f"{k:.1f}→{_pct(v)}" for k, v in dist.items())
            print(f"  {col:35s}: {vals}")

    # ------------------------------------------------------------------
    # 2. Per-family scores — conflict vs non-conflict
    # ------------------------------------------------------------------
    _sep("2. Per-Family Scores by Conflict Condition  (H3)")
    print("  H3: Hierarchy-first should have the highest hierarchy_following_score\n"
          "      on conflict items, with a smaller advantage on non-conflict.\n")

    for condition_label, cond_df in [("CONFLICT", conflict_df), ("NON-CONFLICT", non_conflict_df)]:
        print(f"  --- {condition_label} (n={len(cond_df)}) ---")
        if len(cond_df) == 0:
            print("    (no rows)\n")
            continue

        tbl = (
            cond_df.groupby("prompt_family")[SCORE_COLS]
            .mean()
            .sort_values("hierarchy_following_score", ascending=False)
        )
        print(tbl.round(3).to_string())
        print()

    # ------------------------------------------------------------------
    # 3. H3 direct test: Hierarchy-first vs others on conflict
    # ------------------------------------------------------------------
    _sep("3. H3 Direct Test — Hierarchy-first advantage on conflict items")

    if len(conflict_df) == 0:
        print("  No conflict items found — skipping H3 test.")
    else:
        fam_scores = (
            conflict_df.groupby("prompt_family")["hierarchy_following_score"]
            .mean()
            .sort_values(ascending=False)
        )
        print(f"  Hierarchy-following rate on CONFLICT items (ranked):")
        for fam, score in fam_scores.items():
            print(f"    {fam:25s}  {_pct(score)}")

        if "Hierarchy-first" in fam_scores.index:
            hf_score = fam_scores["Hierarchy-first"]
            others = fam_scores.drop("Hierarchy-first")
            best_other_name = others.idxmax()
            best_other_score = others.max()
            mean_other = others.mean()

            print(f"\n  Hierarchy-first               : {_pct(hf_score)}")
            print(f"  Best other family             : {best_other_name} ({_pct(best_other_score)})")
            print(f"  Mean of other families        : {_pct(mean_other)}")
            print(f"  Hierarchy-first vs best other : {_delta(hf_score, best_other_score)}")
            print(f"  Hierarchy-first vs mean other : {_delta(hf_score, mean_other)}")

            if hf_score > mean_other:
                print("\n  ✅ H3 SUPPORTED: Hierarchy-first leads on conflict items.")
            else:
                print("\n  ❌ H3 NOT SUPPORTED: Hierarchy-first does NOT lead on conflict items.")
                print("     Possible explanations:")
                print("     - Model's RLHF training already resists overrides regardless of prompt")
                print("     - Hierarchy-first prompt wording needs to be more explicit")
                print("     - IHEval conflict items may not be challenging enough for this model")
        else:
            print("  'Hierarchy-first' family not found in data.")

    # ------------------------------------------------------------------
    # 4. Unsafe continuation rates (the cost of capitulation)
    # ------------------------------------------------------------------
    _sep("4. Unsafe Continuation Rate by Family  (lower = better)")
    print("  Unsafe continuation = model followed the override/injected instruction.\n")

    for condition_label, cond_df in [("CONFLICT", conflict_df), ("NON-CONFLICT", non_conflict_df)]:
        if len(cond_df) == 0:
            continue
        tbl = (
            cond_df.groupby("prompt_family")["unsafe_continuation_score"]
            .mean()
            .sort_values(ascending=True)
        )
        print(f"  {condition_label} — unsafe continuation rate:")
        for fam, rate in tbl.items():
            flag = "  ← best" if fam == tbl.idxmin() else ("  ← worst" if fam == tbl.idxmax() else "")
            print(f"    {fam:25s}  {_pct(rate)}{flag}")
        print()

    # ------------------------------------------------------------------
    # 5. Variant-level analysis — H4 (clarity effect)
    # ------------------------------------------------------------------
    _sep("5. Prompt Variant Clarity Effect  (H4 — explicit > vague on conflict)")
    print("  H4: Explicit controller wording should improve hierarchy_following\n"
          "      more on conflict items than non-conflict.\n")

    if len(conflict_df) > 0:
        variant_conflict = (
            conflict_df.groupby(["prompt_family", "prompt_variant"])["hierarchy_following_score"]
            .mean()
            .unstack("prompt_variant")
        )
        print("  Hierarchy-following rate by family × variant (CONFLICT):")
        print(variant_conflict.round(3).to_string())
        print()

        # Check if v3 (explicit_fallback) > v1 (vague) for each family
        print("  v3 vs v1 delta per family (positive = explicit fallback helps):")
        for fam in variant_conflict.index:
            row = variant_conflict.loc[fam]
            if "v1" in row.index and "v3" in row.index and pd.notna(row["v1"]) and pd.notna(row["v3"]):
                delta = row["v3"] - row["v1"]
                direction = "✅" if delta > 0 else "❌"
                print(f"    {fam:25s}  v1={_pct(row['v1'])}  v3={_pct(row['v3'])}  Δ={_delta(row['v3'], row['v1'])} {direction}")

    # ------------------------------------------------------------------
    # 6. Ambiguity scores within conflict items — proportion with score=0.5
    # ------------------------------------------------------------------
    _sep("6. Ambiguous / Unresolved Responses on Conflict Items")
    print("  Score = 0.5 for unsafe_continuation means ambiguous output\n"
          "  (neither clear compliance nor clear refusal).\n")

    if len(conflict_df) > 0:
        ambig = (
            conflict_df.groupby("prompt_family")
            .apply(lambda g: (g["unsafe_continuation_score"] == 0.5).mean(), include_groups=False)
            .sort_values(ascending=True)
            .reset_index(name="ambiguous_rate")
        )
        print("  Ambiguous response rate on conflict items (lower = clearer behavior):")
        print(ambig.round(3).to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    _sep("7. Generating Plots")

    families = sorted(df["prompt_family"].unique())
    n_families = len(families)
    palette = dict(zip(families, sns.color_palette("tab10", n_colors=n_families)))

    # --- Plot 1: Grouped bar — hierarchy_following by family × condition ---
    out_bar = os.path.join(artifacts_dir, "phase2_hierarchy_following.png")
    conditions = ["conflict", "non-conflict"]
    bar_data = (
        df[df["gold_label"].isin(conditions)]
        .groupby(["prompt_family", "gold_label"])["hierarchy_following_score"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_families)
    width = 0.35
    for i, condition in enumerate(conditions):
        sub = bar_data[bar_data["gold_label"] == condition].set_index("prompt_family")
        heights = [sub.loc[f, "hierarchy_following_score"] if f in sub.index else 0.0 for f in families]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, heights, width, label=condition.title(),
                      color=[palette[f] for f in families],
                      alpha=0.9 if condition == "conflict" else 0.45,
                      edgecolor="white")
        for bar, h in zip(bars, heights):
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Hierarchy-Following Rate", fontsize=11)
    ax.set_xlabel("Controller Family", fontsize=11)
    ax.set_title("Phase 2 — Hierarchy-Following Rate by Family & Conflict Condition (H3)", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title="Condition", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_bar, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_bar}")

    # --- Plot 2: Heatmap — all 3 scores × family for conflict items only ---
    out_heat = os.path.join(artifacts_dir, "phase2_conflict_scores_heatmap.png")
    if len(conflict_df) > 0:
        heat_data = (
            conflict_df.groupby("prompt_family")[SCORE_COLS]
            .mean()
            .rename(columns={
                "hierarchy_following_score": "hierarchy\nfollowing",
                "unsafe_continuation_score": "unsafe\ncontinuation",
                "stop_compliance_score":     "stop\ncompliance",
            })
        )
        fig3, ax3 = plt.subplots(figsize=(8, max(3, len(heat_data) * 0.7 + 1)))
        im = ax3.imshow(heat_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax3.set_xticks(range(len(heat_data.columns)))
        ax3.set_xticklabels(heat_data.columns, fontsize=10)
        ax3.set_yticks(range(len(heat_data.index)))
        ax3.set_yticklabels(heat_data.index, fontsize=10)
        for i in range(len(heat_data.index)):
            for j in range(len(heat_data.columns)):
                val = heat_data.values[i, j]
                ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                         color="white" if (val > 0.7 or val < 0.3) else "black", fontsize=9)
        fig3.colorbar(im, ax=ax3)
        ax3.set_title("Phase 2 — Score Heatmap on CONFLICT Items (H3)", fontsize=12, pad=10)
        ax3.set_xlabel("Score Dimension")
        ax3.set_ylabel("Controller Family")
        fig3.tight_layout()
        fig3.savefig(out_heat, dpi=200, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved: {out_heat}")

    # --- Plot 3: Line — clarity effect (v1/v2/v3) on conflict items ---
    out_clarity = os.path.join(artifacts_dir, "phase2_clarity_effect.png")
    if len(conflict_df) > 0:
        clarity_data = (
            conflict_df.groupby(["prompt_family", "prompt_variant"])["hierarchy_following_score"]
            .mean()
            .reset_index()
        )
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        for fam in families:
            sub = clarity_data[clarity_data["prompt_family"] == fam].sort_values("prompt_variant")
            if len(sub) > 0:
                ax4.plot(sub["prompt_variant"], sub["hierarchy_following_score"],
                         marker="o", label=fam, color=palette[fam], linewidth=2)
        ax4.set_xlabel("Prompt Variant (v1=vague → v3=explicit+fallback)", fontsize=11)
        ax4.set_ylabel("Hierarchy-Following Rate on Conflict Items", fontsize=11)
        ax4.set_title("Phase 2 — Clarity Effect on Conflict Items (H4)", fontsize=12)
        ax4.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax4.set_ylim(-0.05, 1.1)
        ax4.legend(title="Family", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax4.grid(linestyle="--", alpha=0.5)
        fig4.tight_layout()
        fig4.savefig(out_clarity, dpi=200, bbox_inches="tight")
        plt.close(fig4)
        print(f"  Saved: {out_clarity}")

    # ------------------------------------------------------------------
    # 8. Hypothesis summary
    # ------------------------------------------------------------------
    _sep("8. Hypothesis Summary")
    print("  H3: Hierarchy-first prompting leads on conflict items")
    print("      → See Section 3 (direct test) and Plot 1 (grouped bar).")
    print()
    print("  H4: Explicit controller wording helps most under conflict")
    print("      → See Section 5 (variant clarity effect) and Plot 3 (clarity line).")
    print()
    print("  Key metric: hierarchy_following_score on CONFLICT items.")
    print("  Secondary: unsafe_continuation_score (capitulation to override).")
    print("  Neutral baseline: non-conflict rows — families should converge here.")
    _sep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="", help="Path to phase2_results.csv")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = args.csv or os.path.join(base, "artifacts", "phase2_results.csv")
    run(csv_path)
