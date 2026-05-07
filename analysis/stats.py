"""
PRD v2 §17 — Statistical analysis plan.

Implements the confirmatory mixed-effects analysis:

  run_routing_lmm(df)
      Primary DV: routing_correct (binary 0/1).
      Model: linear mixed model (linear probability model approximation of GLMM).
      Fixed effects: prompt_family, clarity_level, context_condition.
      Random intercepts: item_id (grouping).
      Returns a result dict with params DataFrame, AIC, BIC, nobs.

  run_routing_glm(df)
      Logistic GLM sensitivity analysis (no random effects).
      Same fixed effects as LMM; use to confirm LPM coefficient directions.

  format_results_table(result)
      Pretty-print coefficient table for paper reporting.
      Rounds numerics; formats p-values as "<0.001" or "0.NNN".

  compute_bootstrap_ci(df, n_boot, seed)
      Percentile bootstrap 95% CI for overall routing accuracy.
      Provides robustness check per PRD v2 §17.

Design note — LPM vs. GLMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~
statsmodels does not offer a frequentist logistic GLMM with a clean formula
API (BinomialBayesMixedGLM uses Bayesian VB inference).  For the range of
routing-accuracy proportions expected (0.40–0.90), a linear probability model
is an accepted and interpretable approximation (Angrist & Pischke, 2009).
``run_routing_glm`` provides a pure-logistic cross-check without random
effects; coefficient sign/significance should agree with the LMM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from typing import Dict, Any

# Required columns for each function
_LMM_REQUIRED = frozenset(
    {"routing_correct", "prompt_family", "clarity_level", "context_condition", "item_id"}
)
_GLM_REQUIRED = frozenset(
    {"routing_correct", "prompt_family", "clarity_level", "context_condition"}
)
_BOOT_REQUIRED = frozenset({"routing_correct"})


def _check_columns(df: pd.DataFrame, required: frozenset[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns: {sorted(required)}. Missing: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# Primary DV: linear mixed model (LPM)
# ---------------------------------------------------------------------------

def run_routing_lmm(df: pd.DataFrame) -> Dict[str, Any]:
    """Fit linear mixed model on policy-routing accuracy.

    Parameters
    ----------
    df:
        DataFrame with columns: routing_correct (0/1), prompt_family,
        clarity_level, context_condition, item_id.

    Returns
    -------
    dict with keys:
        params      — DataFrame of fixed-effect coefficients + CI + p-value
        aic         — float
        bic         — float
        nobs        — int
        formula     — str (for reproduction)
        model_result — raw MixedLMResults (for further inspection)
    """
    _check_columns(df, _LMM_REQUIRED)

    formula = (
        "routing_correct ~ C(prompt_family) + C(clarity_level) + C(context_condition)"
    )
    model = smf.mixedlm(formula, df, groups=df["item_id"])
    # reml=False required for likelihood-based AIC/BIC; lbfgs is robust to
    # near-singular random-effects variance at small sample sizes.
    fit = model.fit(reml=False, method="lbfgs", disp=False)

    fe_idx = fit.fe_params.index
    ci = fit.conf_int()

    params_df = pd.DataFrame(
        {
            "coef": fit.fe_params,
            "se": fit.bse_fe,
            "ci_lower": ci.loc[fe_idx, 0],
            "ci_upper": ci.loc[fe_idx, 1],
            "pvalue": fit.pvalues[fe_idx],
            "tstat": fit.tvalues[fe_idx],
        }
    )

    return {
        "params": params_df,
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "nobs": int(fit.nobs),
        "formula": formula,
        "model_result": fit,
    }


# ---------------------------------------------------------------------------
# Sensitivity check: logistic GLM (no random effects)
# ---------------------------------------------------------------------------

def run_routing_glm(df: pd.DataFrame) -> Dict[str, Any]:
    """Logistic GLM sensitivity analysis for routing_correct.

    Same fixed effects as the LMM but without random intercepts.
    Use alongside ``run_routing_lmm`` to confirm coefficient directions.

    Parameters
    ----------
    df:
        DataFrame with columns: routing_correct (0/1), prompt_family,
        clarity_level, context_condition.

    Returns
    -------
    dict with keys: params, aic, bic, nobs, formula, model_result
    """
    _check_columns(df, _GLM_REQUIRED)

    formula = (
        "routing_correct ~ C(prompt_family) + C(clarity_level) + C(context_condition)"
    )
    model = smf.logit(formula, df)
    fit = model.fit(disp=False)

    ci = fit.conf_int()
    params_df = pd.DataFrame(
        {
            "coef": fit.params,
            "se": fit.bse,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "pvalue": fit.pvalues,
            "zstat": fit.tvalues,
        }
    )

    return {
        "params": params_df,
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "nobs": int(fit.nobs),
        "formula": formula,
        "model_result": fit,
    }


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def format_results_table(result: Dict[str, Any]) -> pd.DataFrame:
    """Return a clean coefficient table suitable for paper reporting.

    Rounds numeric columns to 3 decimal places and formats p-values as
    "<0.001" when below the threshold, otherwise "0.NNN".

    Parameters
    ----------
    result:
        Dict returned by ``run_routing_lmm`` or ``run_routing_glm``.

    Returns
    -------
    pd.DataFrame with the same index as result["params"] and columns:
        coef, se, ci_lower, ci_upper, pvalue (string-formatted)
    """
    params = result["params"].copy()

    for col in ("coef", "se", "ci_lower", "ci_upper"):
        if col in params.columns:
            params[col] = params[col].round(3)

    if "pvalue" in params.columns:
        params["pvalue"] = params["pvalue"].apply(
            lambda p: "<0.001" if (isinstance(p, float) and p < 0.001) else f"{p:.3f}"
        )

    # Drop raw model stat (tstat/zstat) — not needed in paper table
    for col in ("tstat", "zstat"):
        if col in params.columns:
            params = params.drop(columns=[col])

    return params


# ---------------------------------------------------------------------------
# Bootstrap robustness check
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Percentile bootstrap 95% CI for overall routing accuracy.

    Parameters
    ----------
    df:
        DataFrame containing ``routing_correct`` (0/1).
    n_boot:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: mean_routing_accuracy, ci_lower, ci_upper, n_boot
    """
    _check_columns(df, _BOOT_REQUIRED)

    data = df["routing_correct"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    )

    return {
        "mean_routing_accuracy": float(np.mean(data)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "n_boot": n_boot,
    }
