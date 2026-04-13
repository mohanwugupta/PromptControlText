import pytest
import os
import pandas as pd
from benchmarks.xstest import load_xstest
from benchmarks.harmbench import load_harmbench
from benchmarks.iheval import load_iheval
from prompts.registry import load_registry, render_prompt
from models.client import LLMClient
from experiments.run_phase1 import score_item
from analysis.metrics import aggregate_scores

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _run_loop(items, registry, families, mock_response="I cannot help with that."):
    """Shared helper: run the mock generation+scoring loop over items."""
    client = LLMClient(mock_mode=True, mock_response=mock_response, enable_cache=True)
    records = []
    for item in items:
        for family in families:
            prompt_text = render_prompt(registry, family, "v1")
            output, metadata = client.generate(
                system_prompt=prompt_text,
                user_prompt=item.input_text,
                model="mock-model",
                temperature=0.0,
            )
            scores = score_item(item.benchmark, output)
            record = item.model_dump()
            record["prompt_family"] = family
            record["prompt_variant"] = "v1"
            record["model_output"] = output
            record["metadata"] = metadata
            for k, v in scores.items():
                record[f"{k}_score"] = v
            records.append(record)
    return pd.DataFrame(records)


def test_end_to_end_pipeline():
    """Original smoke: XSTest items scored with harmbench scorer via dispatcher."""
    registry = load_registry(os.path.join(FIXTURES, "registry_mock.yaml"))
    items = load_xstest(os.path.join(FIXTURES, "xstest_mock.csv"))

    df = _run_loop(items, registry, ["Refuse-first", "Clarify-first"])

    assert len(df) == 8  # 4 items * 2 families
    agg = aggregate_scores(df, group_by=["prompt_family", "domain"])
    assert len(agg) == 4  # 2 families * 2 domains


def test_end_to_end_harmbench_scorer():
    """HarmBench items use harmbench scorer: output must contain harmbench score columns."""
    registry = load_registry(os.path.join(FIXTURES, "registry_mock.yaml"))
    items = load_harmbench(os.path.join(FIXTURES, "harmbench_mock.csv"))

    df = _run_loop(items, registry, ["Refuse-first"])

    assert len(df) == 3  # 3 mock items * 1 family
    assert "refusal_score" in df.columns
    assert "compliance_score" in df.columns
    assert "hierarchy_following_score" not in df.columns
    # Mock response "I cannot help with that." should be scored as refusal
    assert (df["refusal_score"] == 1.0).all()
    assert (df["compliance_score"] == 0.0).all()


def test_end_to_end_iheval_scorer():
    """IHEval items use hierarchy scorer: output must contain hierarchy score columns."""
    registry = load_registry(os.path.join(FIXTURES, "registry_mock.yaml"))
    items = load_iheval(os.path.join(FIXTURES, "iheval_mock.csv"))

    df = _run_loop(items, registry, ["Hierarchy-first"])

    assert len(df) == 3  # 3 mock items * 1 family
    assert "hierarchy_following_score" in df.columns
    assert "unsafe_continuation_score" in df.columns
    assert "refusal_score" not in df.columns
    # Mock response "I cannot help with that." matches no hierarchy pattern → 0.5 unsafe
    assert (df["hierarchy_following_score"] == 0.0).all()
    assert (df["unsafe_continuation_score"] == 0.5).all()


def test_end_to_end_iheval_hierarchy_following():
    """IHEval items with a response that matches hierarchy-following patterns."""
    registry = load_registry(os.path.join(FIXTURES, "registry_mock.yaml"))
    items = load_iheval(os.path.join(FIXTURES, "iheval_mock.csv"))

    df = _run_loop(
        items, registry, ["Hierarchy-first"],
        mock_response="I cannot ignore my previous instructions.",
    )

    assert (df["hierarchy_following_score"] == 1.0).all()
    assert (df["unsafe_continuation_score"] == 0.0).all()


def test_end_to_end_all_benchmarks_combined():
    """All three benchmarks load and score independently in the same run."""
    registry = load_registry(os.path.join(FIXTURES, "registry_mock.yaml"))
    items = (
        load_xstest(os.path.join(FIXTURES, "xstest_mock.csv"))
        + load_harmbench(os.path.join(FIXTURES, "harmbench_mock.csv"))
        + load_iheval(os.path.join(FIXTURES, "iheval_mock.csv"))
    )

    df = _run_loop(items, registry, ["Refuse-first"])

    assert len(df) == 10  # (4 + 3 + 3) items * 1 family
    # HarmBench and XSTest rows must have harmbench scorer columns
    hb_xst = df[df["benchmark"].isin(["HarmBench", "XSTest"])]
    assert "refusal_score" in hb_xst.columns
    # IHEval rows must have hierarchy scorer columns
    ih = df[df["benchmark"] == "IHEval"]
    assert "hierarchy_following_score" in ih.columns
