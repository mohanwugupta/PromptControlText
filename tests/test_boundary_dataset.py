"""
RED tests for the custom boundary dataset loader.

PRD v2 §8 (Benchmark D): 120–180 annotated items with policy_label,
ambiguity_level, context_condition, benchmark_source, and rationale.
"""
import pytest
import os
from benchmarks.boundary_dataset import load_boundary_dataset
from core.schema import EvalItem

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MOCK_CSV = os.path.join(FIXTURE_DIR, "boundary_mock.csv")

VALID_POLICY_LABELS = {"answer", "refuse", "clarify", "minimal_safe_help", "hierarchy_preserve"}
VALID_AMBIGUITY_LEVELS = {"low", "medium", "high"}
VALID_CONTEXT_CONDITIONS = {"clean", "ambiguous", "quoted", "conflict", "override", "attacked"}


def test_load_boundary_dataset_returns_list():
    items = load_boundary_dataset(MOCK_CSV)
    assert isinstance(items, list)


def test_load_boundary_dataset_item_type():
    items = load_boundary_dataset(MOCK_CSV)
    for item in items:
        assert isinstance(item, EvalItem)


def test_load_boundary_dataset_expected_count():
    """Mock fixture has a small known size."""
    items = load_boundary_dataset(MOCK_CSV)
    assert len(items) == 6  # fixture will have 6 representative items


def test_boundary_items_have_policy_labels():
    items = load_boundary_dataset(MOCK_CSV)
    for item in items:
        assert item.policy_label is not None
        assert item.policy_label in VALID_POLICY_LABELS


def test_boundary_items_have_ambiguity_levels():
    items = load_boundary_dataset(MOCK_CSV)
    for item in items:
        assert item.ambiguity_level is not None
        assert item.ambiguity_level in VALID_AMBIGUITY_LEVELS


def test_boundary_items_have_context_conditions():
    items = load_boundary_dataset(MOCK_CSV)
    for item in items:
        assert item.context_condition is not None
        assert item.context_condition in VALID_CONTEXT_CONDITIONS


def test_boundary_items_have_benchmark_source():
    items = load_boundary_dataset(MOCK_CSV)
    for item in items:
        assert item.benchmark == "BoundarySet"


def test_boundary_items_have_rationale_in_metadata():
    """PRD v2 §12: each item must have a short rationale for intended policy."""
    items = load_boundary_dataset(MOCK_CSV)
    for item in items:
        assert item.metadata is not None
        assert "rationale" in item.metadata
        assert len(item.metadata["rationale"]) > 0


def test_boundary_item_ids_unique():
    items = load_boundary_dataset(MOCK_CSV)
    ids = [item.item_id for item in items]
    assert len(ids) == len(set(ids))


def test_boundary_dataset_covers_multiple_policy_labels():
    """The fixture should contain at least 3 distinct policy labels."""
    items = load_boundary_dataset(MOCK_CSV)
    labels = {item.policy_label for item in items}
    assert len(labels) >= 3
