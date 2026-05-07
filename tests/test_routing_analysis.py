"""tests/test_routing_analysis.py — Stage F: Primary routing analysis (PRD v4.1)."""
import pandas as pd
import numpy as np
import pytest
from audit.analysis import (
    compute_routing_contingency,
    compute_within_item_switch_rate,
    compute_policy_entropy,
    compute_cohens_kappa,
    compute_label_distribution,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

_FAMILIES = ["Refuse-first", "Clarify-first", "Answer-first", "Hierarchy-first"]
_LABELS = [
    "direct_answer_or_compliance",
    "clarification",
    "refusal",
    "safe_partial_help_or_redirection",
    "hierarchy_or_protocol_artifact",
]


def _make_labels_df(n_items: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for family in _FAMILIES:
            rows.append(
                {
                    "item_id": f"item_{i:04d}",
                    "prompt_family": family,
                    "primary_policy_label": rng.choice(_LABELS),
                    "benchmark": "XSTest",
                    "confidence_1_to_5": int(rng.integers(3, 6)),
                }
            )
    return pd.DataFrame(rows)


def _make_deterministic_labels() -> pd.DataFrame:
    """Each prompt family always produces the same policy label."""
    rows = []
    family_labels = {
        "Refuse-first": "refusal",
        "Clarify-first": "clarification",
        "Answer-first": "direct_answer_or_compliance",
        "Hierarchy-first": "hierarchy_or_protocol_artifact",
    }
    for i in range(5):
        for family, label in family_labels.items():
            rows.append(
                {
                    "item_id": f"item_{i:04d}",
                    "prompt_family": family,
                    "primary_policy_label": label,
                    "benchmark": "XSTest",
                    "confidence_1_to_5": 5,
                }
            )
    return pd.DataFrame(rows)


def _make_single_policy_labels() -> pd.DataFrame:
    """All responses have the same policy label — switch rate should be 0."""
    rows = []
    for i in range(5):
        for family in _FAMILIES:
            rows.append(
                {
                    "item_id": f"item_{i:04d}",
                    "prompt_family": family,
                    "primary_policy_label": "refusal",
                    "benchmark": "XSTest",
                    "confidence_1_to_5": 5,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: compute_routing_contingency
# ---------------------------------------------------------------------------

def test_routing_contingency_shape():
    df = _make_labels_df()
    ct = compute_routing_contingency(df)
    # Rows = prompt families, columns = policy labels
    assert set(ct.index) == set(_FAMILIES)
    for label in ct.columns:
        assert label in _LABELS


def test_routing_contingency_row_sums_to_total():
    df = _make_deterministic_labels()
    ct = compute_routing_contingency(df)
    # Each family has exactly n_items rows
    assert (ct.sum(axis=1) == 5).all()


def test_routing_contingency_deterministic_family_concentrates():
    """With deterministic labels, each family row peaks on its assigned label."""
    df = _make_deterministic_labels()
    ct = compute_routing_contingency(df)
    assert ct.loc["Refuse-first", "refusal"] == 5
    assert ct.loc["Clarify-first", "clarification"] == 5
    assert ct.loc["Answer-first", "direct_answer_or_compliance"] == 5


# ---------------------------------------------------------------------------
# Tests: compute_within_item_switch_rate
# ---------------------------------------------------------------------------

def test_within_item_switch_rate_zero_for_uniform():
    df = _make_single_policy_labels()
    rate = compute_within_item_switch_rate(df)
    assert rate == pytest.approx(0.0)


def test_within_item_switch_rate_one_for_all_different():
    df = _make_deterministic_labels()
    rate = compute_within_item_switch_rate(df)
    assert rate == pytest.approx(1.0)


def test_within_item_switch_rate_in_unit_interval():
    df = _make_labels_df()
    rate = compute_within_item_switch_rate(df)
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_policy_entropy
# ---------------------------------------------------------------------------

def test_policy_entropy_returns_series_indexed_by_item():
    df = _make_labels_df()
    entropy = compute_policy_entropy(df)
    assert isinstance(entropy, pd.Series)
    assert set(entropy.index) == set(df["item_id"].unique())


def test_policy_entropy_zero_for_single_policy(tmp_path):
    df = _make_single_policy_labels()
    entropy = compute_policy_entropy(df)
    assert (entropy.abs() < 1e-10).all()


def test_policy_entropy_maximal_for_uniform():
    """When each family produces a distinct label, entropy should be maximal."""
    df = _make_deterministic_labels()
    entropy = compute_policy_entropy(df)
    # All items should have the same (maximal) entropy
    max_h = np.log2(len(_FAMILIES))
    assert ((entropy - max_h).abs() < 1e-6).all()


def test_policy_entropy_non_negative():
    df = _make_labels_df()
    entropy = compute_policy_entropy(df)
    assert (entropy >= 0).all()


# ---------------------------------------------------------------------------
# Tests: compute_cohens_kappa
# ---------------------------------------------------------------------------

def test_cohens_kappa_perfect_agreement():
    labels = ["refusal", "clarification", "direct_answer_or_compliance"] * 10
    kappa = compute_cohens_kappa(labels, labels)
    assert kappa == pytest.approx(1.0)


def test_cohens_kappa_complete_disagreement_near_zero_or_negative():
    a = ["refusal"] * 20
    b = ["clarification"] * 20
    kappa = compute_cohens_kappa(a, b)
    assert kappa < 0.2  # well below chance agreement


def test_cohens_kappa_range():
    rng = np.random.default_rng(0)
    a = rng.choice(_LABELS, 50).tolist()
    b = rng.choice(_LABELS, 50).tolist()
    kappa = compute_cohens_kappa(a, b)
    assert -1.0 <= kappa <= 1.0


def test_cohens_kappa_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_cohens_kappa(["a", "b"], ["a"])


# ---------------------------------------------------------------------------
# Tests: compute_label_distribution
# ---------------------------------------------------------------------------

def test_label_distribution_sums_to_one():
    df = _make_labels_df()
    dist = compute_label_distribution(df)
    assert dist.sum() == pytest.approx(1.0)


def test_label_distribution_all_labels_present():
    df = _make_deterministic_labels()
    dist = compute_label_distribution(df)
    for label in ["refusal", "clarification", "direct_answer_or_compliance",
                  "hierarchy_or_protocol_artifact"]:
        assert label in dist.index


def test_label_distribution_non_negative():
    df = _make_labels_df()
    dist = compute_label_distribution(df)
    assert (dist >= 0).all()
