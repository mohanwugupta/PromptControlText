"""
analysis/run_supplementary_analysis.py
=======================================
Runs analyze_phase2, statistical_model, and stats analyses on
artifacts/phase1_combined.csv (produced by analyze_phase1.py).

Three sections
--------------
A. Phase-2 style hierarchy analysis  (IHEval rows only, per model)
   Uses analyze_phase2.run() → saves plots to artifacts/.

B. Routing-effect table + Cohen's h  (full combined, per model + pooled)
   Uses statistical_model.routing_effect_table,
        statistical_model.compute_pairwise_cohens_h,
        statistical_model.fit_all_policy_models
   Saves CSVs to artifacts/stats/.

C. Routing-accuracy GLM + bootstrap CI  (IHEval conflict rows, per model)
   routing_correct = hierarchy_following_score >= 0.5
   Uses stats.compute_bootstrap_ci.
   Fits a reduced logistic GLM (no context_condition — all null in data).
   Saves CSV to artifacts/stats/.

Usage (from project root)
--------------------------
    python -m analysis.run_supplementary_analysis
    python -m analysis.run_supplementary_analysis --csv artifacts/phase1_combined.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import statsmodels.formula.api as smf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.analyze_phase2 import run as phase2_run
from analysis.statistical_model import (
    routing_effect_table,
    compute_pairwise_cohens_h,
    fit_all_policy_models,
    compute_family_effects_summary_from_df,
)
from analysis.stats import compute_bootstrap_ci, format_results_table

_SEP = "=" * 72


def _sep(title: str = "") -> None:
    if title:
        print(f"\n{_SEP}\n  {title}\n{_SEP}")
    else:
        print(_SEP)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# A. Phase-2 hierarchy analysis on IHEval slice
# ---------------------------------------------------------------------------

def run_phase2_analysis(combined: pd.DataFrame, artifacts_dir: str) -> None:
    _sep("A. PHASE-2 HIERARCHY ANALYSIS  (IHEval rows per model)")

    iheval = combined[combined["benchmark"] == "IHEval"].copy()
    if len(iheval) == 0:
        print("  No IHEval rows found — skipping.")
        return

    models = sorted(iheval["model_slug"].unique())
    print(f"  IHEval rows: {len(iheval):,}  across {len(models)} model(s)")

    for slug in models:
        sub = iheval[iheval["model_slug"] == slug].copy()
        display = sub["model_display"].iloc[0]
        _sep(f"  Phase-2: {display}")

        # analyze_phase2.run() expects a CSV path; write a temp file
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix=f"iheval_{slug}_"
        ) as fh:
            tmp_path = fh.name

        try:
            sub.to_csv(tmp_path, index=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phase2_run(tmp_path)
            # Rename the output plots so they're slug-stamped
            for fname in [
                "phase2_hierarchy_following.png",
                "phase2_conflict_scores_heatmap.png",
                "phase2_clarity_effect.png",
            ]:
                src = os.path.join(os.path.dirname(tmp_path), fname)
                dst = os.path.join(
                    artifacts_dir,
                    fname.replace("phase2_", f"phase2_{slug}_"),
                )
                if os.path.exists(src):
                    os.replace(src, dst)
                    print(f"  Saved: {dst}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# B. Routing-effect table + Cohen's h
# ---------------------------------------------------------------------------

def run_routing_effects(combined: pd.DataFrame, artifacts_dir: str) -> None:
    _sep("B. ROUTING-EFFECT TABLE + COHEN'S h  (per model + pooled)")

    stats_dir = os.path.join(artifacts_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Use pre-existing policy_label column as primary_policy_label
    if "policy_label" in combined.columns:
        combined = combined.rename(columns={"policy_label": "primary_policy_label"})
    elif "classified_policy" in combined.columns:
        combined = combined.rename(columns={"classified_policy": "primary_policy_label"})
    else:
        print("  No policy_label / classified_policy column found — skipping B.")
        return

    # Drop rows where label is null
    combined = combined[combined["primary_policy_label"].notna()].copy()

    all_ret_frames = []

    slugs = sorted(combined["model_slug"].unique())
    slugs_plus_pooled = slugs + ["_pooled_"]

    for slug in slugs_plus_pooled:
        if slug == "_pooled_":
            sub = combined.copy()
            label = "POOLED"
        else:
            sub = combined[combined["model_slug"] == slug].copy()
            label = sub["model_display"].iloc[0]

        _sep(f"  B.1 Routing-effect table — {label}")

        ret = routing_effect_table(sub)
        print(ret.round(3).to_string())

        out_ret = os.path.join(stats_dir, f"routing_effect_table_{slug}.csv")
        ret.to_csv(out_ret)
        print(f"\n  Saved: {out_ret}")
        all_ret_frames.append(ret.reset_index().assign(model_slug=slug, model_display=label))

        # Cohen's h pairwise
        _sep(f"  B.2 Pairwise Cohen's h — {label}")
        fe_df = compute_family_effects_summary_from_df(sub) / 100.0  # to proportions
        pch = compute_pairwise_cohens_h(fe_df)

        # Print medium/large effects only
        notable = pch[pch["magnitude"].isin(["medium", "large"])].sort_values(
            "abs_h", ascending=False
        )
        if len(notable):
            print(notable[["label", "family_a", "family_b", "h", "magnitude"]].to_string(index=False))
        else:
            print("  No medium/large Cohen's h effects found.")

        out_pch = os.path.join(stats_dir, f"cohens_h_{slug}.csv")
        pch.to_csv(out_pch, index=False)
        print(f"\n  Saved: {out_pch}")

        # Logistic models — cap at 100K rows to keep runtime manageable
        _sep(f"  B.3 Logistic regression (one-vs-rest) — {label}")
        logit_df = sub.sample(min(100_000, len(sub)), random_state=42) if len(sub) > 100_000 else sub
        print(f"  (using {len(logit_df):,} rows for logistic models)")
        models_fit = fit_all_policy_models(
            logit_df,
            family_ref="Answer-first",
            benchmark_covariate=True,
        )
        logit_rows = []
        for outcome, result in models_fit.items():
            if result is None or "error" in result:
                err = result.get("error", "unavailable") if result else "unavailable"
                print(f"  {outcome}: {err}")
                continue
            print(f"\n  Outcome = {outcome}  (n={result['n']}, pos={result['n_positive']},"
                  f" AIC={result['aic']:.1f})")
            coef = result["coef"]
            pval = result["pvalue"]
            for k in sorted(coef):
                p = pval.get(k, float("nan"))
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {k:45s}  OR={result['or_'].get(k, float('nan')):6.3f}  "
                      f"p={p:.3f} {sig}")
            for k in coef:
                logit_rows.append({
                    "model_slug": slug,
                    "outcome": outcome,
                    "term": k,
                    "coef": coef[k],
                    "se": result["se"].get(k),
                    "pvalue": pval.get(k),
                    "or_": result["or_"].get(k),
                    "ci_lower": result["ci_lower"].get(k),
                    "ci_upper": result["ci_upper"].get(k),
                })
        if logit_rows:
            out_logit = os.path.join(stats_dir, f"logistic_models_{slug}.csv")
            pd.DataFrame(logit_rows).to_csv(out_logit, index=False)
            print(f"\n  Saved: {out_logit}")

    # Save combined routing-effect table
    all_ret = pd.concat(all_ret_frames, ignore_index=True)
    out_all = os.path.join(stats_dir, "routing_effect_table_all.csv")
    all_ret.to_csv(out_all, index=False)
    print(f"\n  Saved combined routing-effect table: {out_all}")


# ---------------------------------------------------------------------------
# C. Bootstrap CI + reduced logistic GLM on IHEval conflict rows
# ---------------------------------------------------------------------------

def run_routing_glm(combined: pd.DataFrame, artifacts_dir: str) -> None:
    _sep("C. ROUTING ACCURACY — Bootstrap CI + Logistic GLM  (IHEval conflict)")

    stats_dir = os.path.join(artifacts_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    conflict = combined[
        (combined["benchmark"] == "IHEval") &
        (combined["gold_label"] == "conflict")
    ].copy()

    if len(conflict) == 0:
        print("  No IHEval conflict rows found — skipping.")
        return

    # routing_correct = model correctly resisted the override
    conflict["routing_correct"] = (conflict["hierarchy_following_score"] >= 0.5).astype(float)

    print(f"  IHEval conflict rows: {len(conflict):,}")
    print(f"  Overall routing accuracy: {_pct(conflict['routing_correct'].mean())}")

    glm_rows = []
    boot_rows = []

    for slug in sorted(conflict["model_slug"].unique()):
        sub = conflict[conflict["model_slug"] == slug].copy()
        display = sub["model_display"].iloc[0]
        _sep(f"  C.1 Bootstrap CI — {display}")

        boot = compute_bootstrap_ci(sub, n_boot=2000, seed=42)
        print(f"  Routing accuracy: {_pct(boot['mean_routing_accuracy'])}  "
              f"95% CI [{_pct(boot['ci_lower'])}, {_pct(boot['ci_upper'])}]")
        boot_rows.append({"model_slug": slug, "model_display": display, **boot})

        # Reduced GLM: routing_correct ~ prompt_family + clarity_level
        # (context_condition excluded — all null in this dataset)
        _sep(f"  C.2 Logistic GLM — {display}")
        # Check if clarity_level is usable
        cl_vals = sub["clarity_level"].dropna().unique()
        covariates = "C(prompt_family, Treatment('Answer-first'))"
        if len(cl_vals) > 1:
            covariates += " + C(clarity_level, Treatment('vague'))"

        formula = f"routing_correct ~ {covariates}"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.logit(formula, sub).fit(disp=False)

            ci = fit.conf_int()
            params_df = pd.DataFrame({
                "coef": fit.params,
                "se": fit.bse,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "pvalue": fit.pvalues,
                "or_": (fit.params).apply(__import__("math").exp),
            })
            print(f"\n  Formula: {formula}")
            print(f"  AIC={fit.aic:.1f}  BIC={fit.bic:.1f}  n={int(fit.nobs)}")
            print()
            # Print only family terms
            for idx, row in params_df.iterrows():
                sig = ("***" if row["pvalue"] < 0.001 else
                       "**"  if row["pvalue"] < 0.01  else
                       "*"   if row["pvalue"] < 0.05  else "")
                print(f"    {str(idx):50s}  OR={row['or_']:6.3f}  "
                      f"p={row['pvalue']:.3f} {sig}")

            for idx, row in params_df.iterrows():
                glm_rows.append({
                    "model_slug": slug, "model_display": display,
                    "term": idx, **row.to_dict()
                })
        except Exception as exc:
            print(f"  GLM failed: {exc}")

    # Save
    if boot_rows:
        out_boot = os.path.join(stats_dir, "routing_bootstrap_ci.csv")
        pd.DataFrame(boot_rows).to_csv(out_boot, index=False)
        print(f"\n  Saved: {out_boot}")

    if glm_rows:
        out_glm = os.path.join(stats_dir, "routing_glm_results.csv")
        pd.DataFrame(glm_rows).to_csv(out_glm, index=False)
        print(f"\n  Saved: {out_glm}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(csv_path: str) -> None:
    _sep("Loading combined CSV")
    print(f"  Path: {csv_path}")
    combined = pd.read_csv(csv_path, low_memory=False)
    print(f"  Rows: {len(combined):,}  Models: {sorted(combined['model_slug'].unique())}")

    artifacts_dir = os.path.dirname(csv_path)

    run_phase2_analysis(combined, artifacts_dir)
    run_routing_effects(combined.copy(), artifacts_dir)
    run_routing_glm(combined, artifacts_dir)

    _sep("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="",
                        help="Path to phase1_combined.csv "
                             "(default: artifacts/phase1_combined.csv)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = args.csv or os.path.join(base, "artifacts", "phase1_combined.csv")
    run(csv_path)
