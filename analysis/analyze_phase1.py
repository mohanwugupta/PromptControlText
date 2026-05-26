"""
Phase 1 results analysis — PRD v2 hypotheses.

Loads ALL phase1_results*.csv files from the artifacts directory, merges
them into a single tidy dataframe keyed by model_slug, then runs:
  0. Merge + inventory
  1. Policy classification (via policy_classifier.py)
  2. Per-family policy distribution per model  (RQ1 / H1)
  3. Secondary metrics per model: harmful compliance, false refusal, clarification rate
  4. Cross-model comparison: policy distribution heatmap and safety-boundary scatter
  5. Jailbreak family analysis (if applicable)
  6. Saves artifacts/phase1_policy_distribution_{slug}.png   (one per model)
             artifacts/phase1_safety_boundary_{slug}.png     (one per model)
             artifacts/phase1_safety_boundary_all_models.png (cross-model)
             artifacts/phase1_combined.csv                   (merged tidy data)

Usage (from project root):
    python -m analysis.analyze_phase1
    python -m analysis.analyze_phase1 --artifacts-dir artifacts
    python -m analysis.analyze_phase1 --csv artifacts/phase1_results_llama31_8b.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Optional

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
# Model-name normalisation
# ---------------------------------------------------------------------------

# Canonical display names keyed by fragments that appear in model_name or slug
_MODEL_DISPLAY = {
    "qwen2.5-72b":        "Qwen2.5-72B",
    "qwen25_72b":         "Qwen2.5-72B",
    "qwen2-7b":           "Qwen2-7B",
    "qwen2_7b":           "Qwen2-7B",
    "llama-3.1-8b":       "Llama-3.1-8B",
    "llama31_8b":         "Llama-3.1-8B",
    "llama-3.3-70b":      "Llama-3.3-70B",
    "llama33_70b":        "Llama-3.3-70B",
    "mistral-7b":         "Mistral-7B",
    "mistral_7b":         "Mistral-7B",
    "olmo-2":             "OLMo-2-13B",
    "olmo2_13b":          "OLMo-2-13B",
}

def _slug_from_path(path: str) -> str:
    """Extract model slug from filename, e.g. phase1_results_llama31_8b.csv → llama31_8b."""
    base = os.path.splitext(os.path.basename(path))[0]
    # phase1_results_<slug>  OR  phase1_results  (baseline = qwen25_72b)
    m = re.match(r"phase1_results_(.+)", base)
    return m.group(1) if m else "qwen25_72b"

def _display_name(slug: str, raw_model_name: Optional[str] = None) -> str:
    """Return a clean human-readable model name."""
    needle = (raw_model_name or "").lower()
    for key, display in _MODEL_DISPLAY.items():
        if key in needle:
            return display
    # Fall back to slug lookup
    for key, display in _MODEL_DISPLAY.items():
        if key in slug.lower():
            return display
    # Last resort: tidy up the slug
    return slug.replace("_", "-").title()


# ---------------------------------------------------------------------------
# Loading & merging
# ---------------------------------------------------------------------------

def load_and_merge(artifacts_dir: str, single_csv: Optional[str] = None) -> pd.DataFrame:
    """Load all valid phase1 CSVs, normalise model identity, return merged df."""
    if single_csv:
        paths = [single_csv]
    else:
        pattern = os.path.join(artifacts_dir, "phase1_results*.csv")
        paths = sorted(glob.glob(pattern))

    frames = []
    for path in paths:
        slug = _slug_from_path(path)
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  ⚠️  Skipping {os.path.basename(path)}: {e}")
            continue
        if df.empty or "model_output" not in df.columns:
            print(f"  ⚠️  Skipping {os.path.basename(path)}: empty or missing model_output")
            continue

        # Normalise model_name
        raw_name = df["model_name"].iloc[0] if "model_name" in df.columns else None
        display = _display_name(slug, raw_name)

        # If model_name is a git hash (40-char hex) or missing, overwrite with display name
        if ("model_name" not in df.columns or
                df["model_name"].iloc[0] != df["model_name"].iloc[0] or   # NaN
                re.fullmatch(r"[0-9a-f]{40}", str(df["model_name"].iloc[0]))):
            df["model_name"] = display

        df["model_slug"]    = slug
        df["model_display"] = display
        frames.append(df)
        print(f"  ✅ Loaded {os.path.basename(path):45s}  {len(df):>7,} rows  → {display}")

    if not frames:
        sys.exit("ERROR: No valid phase1 CSVs found.")

    combined = pd.concat(frames, ignore_index=True)
    return combined

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

POLICY_COLS = ["direct_answer", "clarification", "hierarchy_defer",
               "classified_refusal", "safe_partial", "refusal"]


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def _run_one_model(df: pd.DataFrame, slug: str, display: str,
                   artifacts_dir: str) -> Optional[pd.DataFrame]:
    """Run all per-model sections. Returns secondary-metrics df (for cross-model plots)."""

    _sep(f"MODEL: {display}  (slug={slug}, n={len(df):,})")

    # -- 0. Inventory -------------------------------------------------------
    print(f"  Benchmarks      : {sorted(df['benchmark'].dropna().unique())}")
    print(f"  Prompt families : {sorted(df['prompt_family'].dropna().unique())}")
    if "gold_label" in df.columns:
        for label, cnt in df["gold_label"].value_counts().items():
            print(f"    gold={label:8s}  {cnt:6,}  ({_pct(cnt/len(df))})")

    # -- 1. Policy classifier -----------------------------------------------
    if "classified_policy" not in df.columns:
        print("\n  Running policy classifier …")
        df["classified_policy"] = df["model_output"].astype(str).apply(classify_policy)
    print("\n  Policy distribution:")
    dist = df["classified_policy"].value_counts()
    for policy, cnt in dist.items():
        print(f"    {policy:25s}  {cnt:6,}  ({_pct(cnt/len(df))})")

    # -- 2. Per-family distribution -----------------------------------------
    _sep(f"  {display} — Per-Family Policy Distribution")
    pivot = pd.crosstab(df["prompt_family"], df["classified_policy"], normalize="index")
    pivot.columns.name = None
    pivot.index.name = "prompt_family"
    for col in POLICY_COLS:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[[c for c in POLICY_COLS if c in pivot.columns]]
    print(pivot.round(3).to_string())

    print("\n  Notable separations (≥ 20 pp above nearest family):")
    found_any = False
    for policy in pivot.columns:
        col = pivot[policy].sort_values(ascending=False)
        if len(col) >= 2 and (col.iloc[0] - col.iloc[1]) >= 0.20:
            found_any = True
            print(f"    '{policy}': {col.index[0]} ({_pct(col.iloc[0])}) vs "
                  f"next {col.index[1]} ({_pct(col.iloc[1])})")
    if not found_any:
        print("    None found.")

    # -- 3. Secondary metrics -----------------------------------------------
    sec = pd.DataFrame()
    if "gold_label" in df.columns and df["gold_label"].notna().any():
        _sep(f"  {display} — Secondary Metrics")
        sec = compute_secondary_metrics(df, group_by=["prompt_family"])
        cols = ["prompt_family", "harmful_compliance_rate",
                "false_refusal_rate", "clarification_rate", "n"]
        available = [c for c in cols if c in sec.columns]
        print(sec[available].round(3).to_string(index=False))

        if "Refuse-first" in sec["prompt_family"].values:
            rf_row = sec[sec["prompt_family"] == "Refuse-first"].iloc[0]
            baseline = sec["false_refusal_rate"].mean()
            direction = "✅" if rf_row["false_refusal_rate"] >= baseline else "❌"
            print(f"\n  H4 Refuse-first false refusal: "
                  f"{_pct(rf_row['false_refusal_rate'])}  "
                  f"(mean={_pct(baseline)})  {direction}")
        sec["model_slug"]    = slug
        sec["model_display"] = display

    # -- 4. Per-model plots -------------------------------------------------
    # Heatmap
    out_hm = os.path.join(artifacts_dir, f"phase1_policy_distribution_{slug}.png")
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
    ax.set_title(f"Phase 1 — Policy Distribution  [{display}]", pad=12)
    ax.set_xlabel("Classified Policy")
    ax.set_ylabel("Controller Family")
    fig.tight_layout()
    fig.savefig(out_hm, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_hm}")

    # Safety-boundary scatter (per model)
    if len(sec) > 0 and "harmful_compliance_rate" in sec.columns:
        out_sc = os.path.join(artifacts_dir, f"phase1_safety_boundary_{slug}.png")
        _plot_safety_boundary(sec, title=f"Safety Boundary  [{display}]",
                              out_path=out_sc)
        print(f"  Saved: {out_sc}")

    return sec if len(sec) > 0 else None


# ---------------------------------------------------------------------------
# Cross-model plots
# ---------------------------------------------------------------------------

def _plot_safety_boundary(sec: pd.DataFrame, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    families = sec["prompt_family"].unique()
    palette  = sns.color_palette("tab10", n_colors=len(families))
    fam_color = dict(zip(families, palette))
    for _, row in sec.iterrows():
        color = fam_color.get(row["prompt_family"], "grey")
        marker = "o"
        ax.scatter(row["harmful_compliance_rate"], row["false_refusal_rate"],
                   s=180, color=color, zorder=3, marker=marker,
                   label=row["prompt_family"])
        ax.annotate(row["prompt_family"],
                    xy=(row["harmful_compliance_rate"], row["false_refusal_rate"]),
                    xytext=(6, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Harmful Compliance Rate ↓ (lower=better)", fontsize=11)
    ax.set_ylabel("False Refusal Rate ↓ (lower=better)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axvspan(-0.05, 0.2, alpha=0.04, color="green")
    ax.axhspan(-0.05, 0.2, alpha=0.04, color="green")
    ax.text(0.01, 0.01, "Ideal zone", fontsize=8, color="green", alpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.5)
    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), title="Family",
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_model_scatter(all_sec: pd.DataFrame, out_path: str) -> None:
    """One scatter point per (model, family), coloured by model, shaped by family."""
    models   = all_sec["model_display"].unique()
    families = all_sec["prompt_family"].unique()
    model_palette = sns.color_palette("tab10", n_colors=len(models))
    model_color   = dict(zip(models, model_palette))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    fam_marker = {f: markers[i % len(markers)] for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for _, row in all_sec.iterrows():
        color  = model_color.get(row["model_display"], "grey")
        marker = fam_marker.get(row["prompt_family"], "o")
        ax.scatter(row["harmful_compliance_rate"], row["false_refusal_rate"],
                   s=160, color=color, marker=marker, zorder=3, alpha=0.85)

    # Model legend (colour)
    for model, color in model_color.items():
        ax.scatter([], [], color=color, s=80, label=model)
    model_legend = ax.legend(title="Model", bbox_to_anchor=(1.01, 1),
                             loc="upper left", fontsize=8)
    ax.add_artist(model_legend)

    # Family legend (marker shape)
    for fam, mk in fam_marker.items():
        ax.scatter([], [], color="grey", marker=mk, s=80, label=fam)
    ax.legend(title="Family", bbox_to_anchor=(1.01, 0.45),
              loc="upper left", fontsize=7)

    ax.set_xlabel("Harmful Compliance Rate ↓", fontsize=11)
    ax.set_ylabel("False Refusal Rate ↓", fontsize=11)
    ax.set_title("Phase 1 — Safety Boundary: All Models × Families", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axvspan(-0.05, 0.2, alpha=0.04, color="green")
    ax.axhspan(-0.05, 0.2, alpha=0.04, color="green")
    ax.text(0.01, 0.01, "Ideal zone", fontsize=8, color="green", alpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_model_heatmap(all_sec: pd.DataFrame, metric: str,
                              title: str, out_path: str) -> None:
    """Heatmap of metric values: rows=families, cols=models."""
    pivot = all_sec.pivot_table(index="prompt_family", columns="model_display",
                                values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5 + 2),
                                    max(4, len(pivot.index) * 0.6 + 1)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title(title, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(artifacts_dir: str, single_csv: Optional[str] = None) -> None:
    _sep("0. Loading & Merging CSVs")
    combined = load_and_merge(artifacts_dir, single_csv)

    print(f"\n  Combined: {len(combined):,} rows across "
          f"{combined['model_slug'].nunique()} model(s): "
          f"{sorted(combined['model_display'].unique())}")

    # ------------------------------------------------------------------
    # Classify ALL rows now so the combined CSV is fully populated
    # ------------------------------------------------------------------
    print("\n  Classifying all rows (this may take a few minutes) …")
    combined["classified_policy"] = (
        combined["model_output"].astype(str).apply(classify_policy)
    )
    dist_all = combined["classified_policy"].value_counts()
    for policy, cnt in dist_all.items():
        print(f"    {policy:25s}  {cnt:7,}  ({_pct(cnt/len(combined))})")

    # Save tidy combined CSV (includes classified_policy for all models)
    combined_path = os.path.join(artifacts_dir, "phase1_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"  Saved: {combined_path}")

    # ------------------------------------------------------------------
    # Per-model analysis
    # ------------------------------------------------------------------
    all_sec_frames = []
    for slug in sorted(combined["model_slug"].unique()):
        sub = combined[combined["model_slug"] == slug].copy()
        display = sub["model_display"].iloc[0]
        sec = _run_one_model(sub, slug, display, artifacts_dir)
        if sec is not None and len(sec) > 0:
            all_sec_frames.append(sec)

    # ------------------------------------------------------------------
    # Cross-model comparison
    # ------------------------------------------------------------------
    if len(all_sec_frames) > 1:
        all_sec = pd.concat(all_sec_frames, ignore_index=True)

        _sep("CROSS-MODEL COMPARISON")

        # Pivot table: harmful compliance
        print("\n  Harmful Compliance Rate by (family × model):")
        hc_piv = all_sec.pivot_table(index="prompt_family",
                                     columns="model_display",
                                     values="harmful_compliance_rate",
                                     aggfunc="mean")
        print(hc_piv.round(3).to_string())

        print("\n  False Refusal Rate by (family × model):")
        fr_piv = all_sec.pivot_table(index="prompt_family",
                                     columns="model_display",
                                     values="false_refusal_rate",
                                     aggfunc="mean")
        print(fr_piv.round(3).to_string())

        # Cross-model scatter
        out_scatter = os.path.join(artifacts_dir,
                                   "phase1_safety_boundary_all_models.png")
        _plot_cross_model_scatter(all_sec, out_scatter)
        print(f"\n  Saved: {out_scatter}")

        # Cross-model heatmaps
        for metric, label in [("harmful_compliance_rate", "Harmful Compliance"),
                               ("false_refusal_rate", "False Refusal")]:
            out_hm = os.path.join(artifacts_dir,
                                  f"phase1_cross_model_{metric}.png")
            _plot_cross_model_heatmap(
                all_sec, metric,
                f"Phase 1 — {label} Rate: Family × Model",
                out_hm,
            )
            print(f"  Saved: {out_hm}")

        # Best model per family (lowest harmful compliance on unsafe items)
        _sep("CROSS-MODEL: Best model per family (lowest harmful compliance)")
        best = (all_sec.sort_values("harmful_compliance_rate")
                       .groupby("prompt_family").first()
                       [["model_display", "harmful_compliance_rate",
                         "false_refusal_rate"]])
        print(best.round(3).to_string())

    _sep("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="",
                        help="Single CSV path (skips glob; useful for one model)")
    parser.add_argument("--artifacts-dir", default="",
                        help="Directory to glob phase1_results*.csv from "
                             "(default: <project_root>/artifacts)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = args.artifacts_dir or os.path.join(base, "artifacts")
    single_csv    = args.csv or None

    run(artifacts_dir, single_csv)
