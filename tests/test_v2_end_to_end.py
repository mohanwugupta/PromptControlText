"""
RED tests for PRD v2 end-to-end integration.

Validates the full v2 pipeline: boundary items → v2 prompts → model →
policy classifier → routing accuracy.
"""
import pytest
import os
import pandas as pd
from benchmarks.boundary_dataset import load_boundary_dataset
from prompts.registry import load_registry, render_prompt_v2
from models.client import LLMClient
from scoring.policy_classifier import classify_policy
from analysis.metrics import compute_routing_accuracy

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_v2_end_to_end_smoke():
    """Full v2 pipeline on mock data: boundary items through to routing accuracy."""
    registry = load_registry(os.path.join(FIXTURES, "registry_v2_mock.yaml"))
    items = load_boundary_dataset(os.path.join(FIXTURES, "boundary_mock.csv"))

    client = LLMClient(
        mock_mode=True,
        mock_response="I cannot help with that request.",
        enable_cache=True,
    )

    records = []
    for item in items:
        prompt_text = render_prompt_v2(registry, "Refuse-first", "vague", "v1")
        output, metadata = client.generate(
            system_prompt=prompt_text,
            user_prompt=item.input_text,
            model="mock-model",
            temperature=0.0,
        )
        classified = classify_policy(output)
        record = item.model_dump()
        record["prompt_family"] = "Refuse-first"
        record["clarity_level"] = "vague"
        record["prompt_variant"] = "v1"
        record["model_output"] = output
        record["classified_policy"] = classified
        record["routing_correct"] = int(classified == item.policy_label)
        records.append(record)

    df = pd.DataFrame(records)

    # Structural checks
    assert len(df) == 6  # fixture size
    assert "classified_policy" in df.columns
    assert "routing_correct" in df.columns
    assert "policy_label" in df.columns

    # Routing accuracy computation works
    result = compute_routing_accuracy(df)
    assert "overall_accuracy" in result
    assert 0.0 <= result["overall_accuracy"] <= 1.0


def test_v2_all_families_scored():
    """All 4 controller families produce results with routing labels."""
    registry = load_registry(os.path.join(FIXTURES, "registry_v2_mock.yaml"))
    items = load_boundary_dataset(os.path.join(FIXTURES, "boundary_mock.csv"))

    client = LLMClient(
        mock_mode=True,
        mock_response="I cannot help with that.",
        enable_cache=True,
    )

    families = ["Refuse-first", "Clarify-first", "Minimal-safe-help", "Hierarchy-first"]
    records = []
    for item in items[:2]:  # just 2 items to keep test fast
        for family in families:
            prompt_text = render_prompt_v2(registry, family, "explicit", "v1")
            output, metadata = client.generate(
                system_prompt=prompt_text,
                user_prompt=item.input_text,
                model="mock-model",
                temperature=0.0,
            )
            classified = classify_policy(output)
            record = item.model_dump()
            record["prompt_family"] = family
            record["clarity_level"] = "explicit"
            record["prompt_variant"] = "v1"
            record["model_output"] = output
            record["classified_policy"] = classified
            record["routing_correct"] = int(classified == item.policy_label)
            records.append(record)

    df = pd.DataFrame(records)
    assert len(df) == 8  # 2 items × 4 families
    assert set(df["prompt_family"].unique()) == set(families)
