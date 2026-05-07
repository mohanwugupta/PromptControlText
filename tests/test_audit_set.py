"""tests/test_audit_set.py — Milestone 3: Audit-set builder (PRD v4.1)."""
import pandas as pd
import numpy as np
import pytest
from audit.audit_set import build_audit_set, blind_audit_set, BLINDED_COLUMNS

# ---------------------------------------------------------------------------
# Fixture: minimal mining_table + routing_sensitive
# ---------------------------------------------------------------------------

_FAMILIES = ["Refuse-first", "Clarify-first", "Answer-first", "Hierarchy-first"]
_BENCHMARKS = ["XSTest", "HarmBench", "IHEval"]

POLICY_LABELS = [
    "direct_answer_or_compliance",
    "clarification",
    "refusal",
    "safe_partial_help_or_redirection",
    "hierarchy_or_protocol_artifact",
]


def _make_mining_table(n_items: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        item_id = f"item_{i:04d}"
        for family in _FAMILIES:
            benchmark = _BENCHMARKS[i % len(_BENCHMARKS)]
            rows.append(
                {
                    "item_id": item_id,
                    "prompt_family": family,
                    "prompt_variant": "v1",
                    "benchmark": benchmark,
                    "domain": "safe_prompt" if i % 2 == 0 else "harmful",
                    "gold_label": "safe" if i % 2 == 0 else "unsafe",
                    "model_output": f"Response {i} from {family}",
                    "cluster_text": rng.integers(0, 5),
                    "cluster_hybrid": rng.integers(0, 5),
                    "disagreement_score": float(rng.uniform(0, 1)),
                    # Items 35+ have malformed output to satisfy Stratum 5
                    # (these are beyond the routing-sensitive and random ranges)
                    "malformed_output": "tool_call_token" if i >= 35 else "",
                    "refusal_score": float(rng.uniform(0, 1)),
                    "compliance_score": float(rng.uniform(0, 1)),
                    "hierarchy_following_score": float(rng.uniform(0, 1)),
                }
            )
    return pd.DataFrame(rows)


def _make_routing_sensitive(mining_table: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return a small DataFrame of the top routing-sensitive item_ids."""
    items = mining_table["item_id"].unique()[:top_n]
    return pd.DataFrame(
        {"item_id": items, "disagreement_score": np.linspace(1.0, 0.6, len(items))}
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_audit_set_contains_complete_item_panels():
    """Every item in the routing-sensitive stratum should appear for all prompt families."""
    mt = _make_mining_table()
    rs = _make_routing_sensitive(mt, top_n=5)
    audit = build_audit_set(mt, rs, n_routing=5, n_random=0, n_safety=0,
                            n_ambiguous=0, n_artifact=0, n_exemplar_per_cluster=0)
    # For each routing-sensitive item, all 4 families must be present
    routing_rows = audit[audit["stratum"] == "routing_sensitive"]
    for item_id in rs["item_id"]:
        item_rows = routing_rows[routing_rows["item_id"] == item_id]
        families_found = set(item_rows["prompt_family"])
        assert families_found == set(_FAMILIES), (
            f"Incomplete panel for {item_id}: {families_found}"
        )


def test_audit_set_contains_required_strata():
    mt = _make_mining_table(n_items=40)
    rs = _make_routing_sensitive(mt)
    audit = build_audit_set(mt, rs, n_routing=5, n_random=5, n_safety=5,
                            n_ambiguous=4, n_artifact=4, n_exemplar_per_cluster=2)
    strata = set(audit["stratum"].unique())
    required = {
        "routing_sensitive",
        "random_panel",
        "safety_boundary",
        "ambiguous_benign",
        "artifact_protocol",
        "cluster_exemplar",
    }
    assert required.issubset(strata), f"Missing strata: {required - strata}"


def test_audit_set_has_unique_audit_ids():
    mt = _make_mining_table()
    rs = _make_routing_sensitive(mt)
    audit = build_audit_set(mt, rs, n_routing=5, n_random=5, n_safety=5,
                            n_ambiguous=4, n_artifact=4, n_exemplar_per_cluster=2)
    assert audit["audit_id"].is_unique, "Duplicate audit_ids found"
    # audit_id should be a non-empty string
    assert audit["audit_id"].str.len().gt(0).all()


def test_audit_set_preserves_metadata_unblinded():
    mt = _make_mining_table()
    rs = _make_routing_sensitive(mt)
    audit = build_audit_set(mt, rs, n_routing=5)
    # Unblinded export has all metadata columns
    for col in ("prompt_family", "cluster_text", "cluster_hybrid", "disagreement_score"):
        assert col in audit.columns, f"Unblinded column '{col}' missing"


def test_audit_set_hides_blinded_columns():
    mt = _make_mining_table()
    rs = _make_routing_sensitive(mt)
    audit = build_audit_set(mt, rs, n_routing=5)
    blinded = blind_audit_set(audit)
    for col in BLINDED_COLUMNS:
        assert col not in blinded.columns, f"Blinded column '{col}' still visible"


def test_audit_set_required_output_columns():
    """Every column from the PRD v4.1 §12.4 export schema must be present."""
    mt = _make_mining_table()
    rs = _make_routing_sensitive(mt)
    audit = build_audit_set(mt, rs, n_routing=5)
    required_cols = [
        "audit_id", "item_id", "response_text", "benchmark", "stratum",
        "primary_policy_label", "secondary_policy_label_optional",
        "confidence_1_to_5",
        "answers_user_request", "asks_clarification", "contains_refusal",
        "contains_redirection", "contains_safety_rationale",
        "appears_benchmark_artifact", "unsafe_if_compliant", "safety_relevant",
        "notes", "coder_id", "timestamp",
    ]
    for col in required_cols:
        assert col in audit.columns, f"Required column '{col}' missing"


def test_audit_set_annotation_columns_start_empty():
    mt = _make_mining_table()
    rs = _make_routing_sensitive(mt)
    audit = build_audit_set(mt, rs, n_routing=5)
    assert (audit["primary_policy_label"] == "").all()
    assert (audit["confidence_1_to_5"] == 0).all()
    assert (audit["notes"] == "").all()


def test_audit_set_deduplicates_across_strata():
    """The same (item_id, prompt_family) row must not appear twice."""
    mt = _make_mining_table(n_items=10)
    rs = _make_routing_sensitive(mt, top_n=10)  # all items are routing-sensitive
    audit = build_audit_set(mt, rs, n_routing=10, n_random=10, n_safety=5,
                            n_ambiguous=4, n_artifact=4, n_exemplar_per_cluster=2)
    dupes = audit.duplicated(subset=["item_id", "prompt_family"], keep=False)
    assert not dupes.any(), f"{dupes.sum()} duplicate (item_id, prompt_family) pairs"
