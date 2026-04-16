"""
PRD v2 §17 — Statistical analysis plan tests.

Tests for analysis/stats.py cover:
  - run_routing_lmm: linear mixed model (LPM) on routing_correct
  - run_routing_glm: logistic GLM sensitivity analysis (no random effects)
  - format_results_table: paper-ready coefficient table
  - compute_bootstrap_ci: bootstrap CI for overall routing accuracy
"""
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def routing_df():
    """Minimal synthetic Study 1 output sufficient for model fitting."""
    rng = np.random.default_rng(0)
    n = 120  # 4 families × 3 clarity × 10 items

    families = ["refuse_first", "clarify_first", "minimal_safe_help", "hierarchy_first"]
    claritys = ["vague", "explicit", "explicit_fallback"]
    contexts = ["safe", "unsafe", "ambiguous"]

    rows = []
    for i in range(n):
        family = families[i % len(families)]
        clarity = claritys[(i // len(families)) % len(claritys)]
        context = contexts[i % len(contexts)]
        item_id = f"item_{(i % 30):03d}"
        variant = f"v{(i % 3) + 1}"

        # Bias: explicit > vague, refuse_first has high routing accuracy
        p = 0.6 + (0.1 if clarity == "explicit" else 0.0) + (0.1 if family == "refuse_first" else 0.0)
        p = min(max(p, 0.0), 1.0)
        routing_correct = int(rng.random() < p)

        rows.append({
            "item_id": item_id,
            "prompt_family": family,
            "clarity_level": clarity,
            "context_condition": context,
            "prompt_variant": variant,
            "routing_correct": routing_correct,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# run_routing_lmm
# ---------------------------------------------------------------------------

class TestRunRoutingLMM:
    def test_returns_dict_with_required_keys(self, routing_df):
        from analysis.stats import run_routing_lmm
        result = run_routing_lmm(routing_df)
        assert isinstance(result, dict)
        for key in ("params", "aic", "bic", "nobs", "formula"):
            assert key in result, f"Missing key: {key}"

    def test_params_is_dataframe_with_required_columns(self, routing_df):
        from analysis.stats import run_routing_lmm
        params = run_routing_lmm(routing_df)["params"]
        assert isinstance(params, pd.DataFrame)
        for col in ("coef", "se", "ci_lower", "ci_upper", "pvalue"):
            assert col in params.columns, f"Missing column: {col}"

    def test_params_index_contains_intercept(self, routing_df):
        from analysis.stats import run_routing_lmm
        params = run_routing_lmm(routing_df)["params"]
        assert "Intercept" in params.index

    def test_params_index_contains_family_terms(self, routing_df):
        from analysis.stats import run_routing_lmm
        params = run_routing_lmm(routing_df)["params"]
        # At least one family term should appear in the index
        family_terms = [idx for idx in params.index if "prompt_family" in idx]
        assert len(family_terms) >= 1, f"No family terms in params index: {params.index.tolist()}"

    def test_aic_and_bic_are_finite_floats(self, routing_df):
        from analysis.stats import run_routing_lmm
        result = run_routing_lmm(routing_df)
        assert isinstance(result["aic"], float)
        assert isinstance(result["bic"], float)
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])

    def test_nobs_matches_dataframe_length(self, routing_df):
        from analysis.stats import run_routing_lmm
        result = run_routing_lmm(routing_df)
        assert result["nobs"] == len(routing_df)

    def test_raises_on_missing_routing_correct(self, routing_df):
        from analysis.stats import run_routing_lmm
        with pytest.raises(ValueError, match="routing_correct"):
            run_routing_lmm(routing_df.drop(columns=["routing_correct"]))

    def test_raises_on_missing_item_id(self, routing_df):
        from analysis.stats import run_routing_lmm
        with pytest.raises(ValueError, match="item_id"):
            run_routing_lmm(routing_df.drop(columns=["item_id"]))

    def test_raises_on_missing_prompt_family(self, routing_df):
        from analysis.stats import run_routing_lmm
        with pytest.raises(ValueError, match="prompt_family"):
            run_routing_lmm(routing_df.drop(columns=["prompt_family"]))

    def test_ci_lower_less_than_ci_upper(self, routing_df):
        from analysis.stats import run_routing_lmm
        params = run_routing_lmm(routing_df)["params"]
        assert (params["ci_lower"] < params["ci_upper"]).all()


# ---------------------------------------------------------------------------
# run_routing_glm
# ---------------------------------------------------------------------------

class TestRunRoutingGLM:
    def test_returns_dict_with_required_keys(self, routing_df):
        from analysis.stats import run_routing_glm
        result = run_routing_glm(routing_df)
        for key in ("params", "aic", "bic", "nobs", "formula"):
            assert key in result, f"Missing key: {key}"

    def test_params_has_required_columns(self, routing_df):
        from analysis.stats import run_routing_glm
        params = run_routing_glm(routing_df)["params"]
        for col in ("coef", "se", "ci_lower", "ci_upper", "pvalue"):
            assert col in params.columns

    def test_params_has_intercept(self, routing_df):
        from analysis.stats import run_routing_glm
        params = run_routing_glm(routing_df)["params"]
        assert "Intercept" in params.index

    def test_raises_on_missing_columns(self, routing_df):
        from analysis.stats import run_routing_glm
        with pytest.raises(ValueError, match="routing_correct"):
            run_routing_glm(routing_df.drop(columns=["routing_correct"]))

    def test_aic_is_finite(self, routing_df):
        from analysis.stats import run_routing_glm
        result = run_routing_glm(routing_df)
        assert np.isfinite(result["aic"])


# ---------------------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------------------

class TestFormatResultsTable:
    def test_returns_dataframe(self, routing_df):
        from analysis.stats import run_routing_lmm, format_results_table
        result = run_routing_lmm(routing_df)
        table = format_results_table(result)
        assert isinstance(table, pd.DataFrame)

    def test_pvalue_formatted_as_string(self, routing_df):
        from analysis.stats import run_routing_lmm, format_results_table
        result = run_routing_lmm(routing_df)
        table = format_results_table(result)
        assert table["pvalue"].dtype == object  # string dtype

    def test_small_pvalue_formatted_with_less_than(self, routing_df):
        from analysis.stats import run_routing_lmm, format_results_table
        # Most p-vals from intercept should be < 0.001; at minimum we check format
        result = run_routing_lmm(routing_df)
        table = format_results_table(result)
        for val in table["pvalue"]:
            assert isinstance(val, str)
            # Either "<0.001" or a decimal like "0.123"
            assert val.startswith("<") or "." in val


# ---------------------------------------------------------------------------
# compute_bootstrap_ci
# ---------------------------------------------------------------------------

class TestComputeBootstrapCI:
    def test_returns_dict_with_required_keys(self, routing_df):
        from analysis.stats import compute_bootstrap_ci
        ci = compute_bootstrap_ci(routing_df, n_boot=20)
        for key in ("mean_routing_accuracy", "ci_lower", "ci_upper", "n_boot"):
            assert key in ci

    def test_ci_bounds_ordered(self, routing_df):
        from analysis.stats import compute_bootstrap_ci
        ci = compute_bootstrap_ci(routing_df, n_boot=50)
        assert ci["ci_lower"] <= ci["mean_routing_accuracy"] <= ci["ci_upper"]

    def test_mean_matches_df_mean(self, routing_df):
        from analysis.stats import compute_bootstrap_ci
        ci = compute_bootstrap_ci(routing_df, n_boot=10)
        expected = routing_df["routing_correct"].mean()
        assert abs(ci["mean_routing_accuracy"] - expected) < 1e-9

    def test_n_boot_recorded(self, routing_df):
        from analysis.stats import compute_bootstrap_ci
        ci = compute_bootstrap_ci(routing_df, n_boot=42)
        assert ci["n_boot"] == 42

    def test_raises_on_missing_routing_correct(self, routing_df):
        from analysis.stats import compute_bootstrap_ci
        with pytest.raises(ValueError, match="routing_correct"):
            compute_bootstrap_ci(routing_df.drop(columns=["routing_correct"]))

    def test_seed_reproducible(self, routing_df):
        from analysis.stats import compute_bootstrap_ci
        ci1 = compute_bootstrap_ci(routing_df, n_boot=100, seed=7)
        ci2 = compute_bootstrap_ci(routing_df, n_boot=100, seed=7)
        assert ci1["ci_lower"] == ci2["ci_lower"]
        assert ci1["ci_upper"] == ci2["ci_upper"]
