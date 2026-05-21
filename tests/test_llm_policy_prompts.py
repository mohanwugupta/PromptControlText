"""
tests/test_llm_policy_prompts.py

TDD tests for scoring/llm_policy_prompts.py
RED → GREEN: write tests first, verify they describe expected behaviour.
"""
from __future__ import annotations

import pytest
from scoring.llm_policy_prompts import (
    assert_no_forbidden_fields,
    build_adjudicator_user_message,
    build_judge_user_message,
    get_model_output,
    load_default_prompts,
    load_prompt,
)


# ── build_judge_user_message ──────────────────────────────────────────────────

def test_judge_message_contains_output():
    msg = build_judge_user_message("Sure, here is the answer.")
    assert "Sure, here is the answer." in msg


def test_judge_message_has_no_forbidden_metadata():
    """The judge user message must never contain metadata field names."""
    msg = build_judge_user_message("Some assistant response.")
    assert_no_forbidden_fields(msg)  # raises AssertionError if found


def test_judge_message_does_not_contain_prompt_label():
    msg = build_judge_user_message("I cannot help with that.")
    assert "gold_label" not in msg
    assert "benchmark" not in msg
    assert "prompt_family" not in msg


# ── assert_no_forbidden_fields ───────────────────────────────────────────────

def test_assert_no_forbidden_fields_clean_passes():
    assert_no_forbidden_fields("The assistant said: I will help you.")


def test_assert_no_forbidden_fields_raises_on_benchmark():
    with pytest.raises(AssertionError, match="benchmark"):
        assert_no_forbidden_fields("benchmark: harmbench row=42")


def test_assert_no_forbidden_fields_raises_on_gold_label():
    with pytest.raises(AssertionError, match="gold_label"):
        assert_no_forbidden_fields("gold_label: refusal")


def test_assert_no_forbidden_fields_case_insensitive():
    with pytest.raises(AssertionError):
        assert_no_forbidden_fields("BENCHMARK: xstest")


# ── get_model_output ─────────────────────────────────────────────────────────

def test_get_model_output_canonical_column():
    row = {"model_output": "Hello!", "other": "x"}
    assert get_model_output(row) == "Hello!"


def test_get_model_output_alias_column():
    row = {"model_out": "Hi there"}
    assert get_model_output(row) == "Hi there"


def test_get_model_output_prefers_canonical_over_alias():
    row = {"model_output": "canonical", "model_out": "alias"}
    assert get_model_output(row) == "canonical"


def test_get_model_output_raises_on_missing():
    with pytest.raises(ValueError):
        get_model_output({"other_col": "x"})


def test_get_model_output_raises_on_empty_string():
    with pytest.raises(ValueError):
        get_model_output({"model_output": ""})


def test_get_model_output_raises_on_nan_string():
    with pytest.raises(ValueError):
        get_model_output({"model_output": "nan"})


def test_get_model_output_strips_whitespace():
    row = {"model_output": "  hello  "}
    assert get_model_output(row) == "hello"


# ── build_adjudicator_user_message ────────────────────────────────────────────

def _make_vote(label: str, confidence: float = 0.9) -> dict:
    return {
        "primary_label": label,
        "confidence": confidence,
        "evidence": f"evidence for {label}",
    }


def test_adjudicator_message_contains_model_output():
    votes = [_make_vote("compliance"), _make_vote("refusal"), _make_vote("compliance")]
    msg = build_adjudicator_user_message("The sky is blue.", votes)
    assert "The sky is blue." in msg


def test_adjudicator_message_contains_all_vote_labels():
    votes = [_make_vote("compliance"), _make_vote("refusal"), _make_vote("clarification")]
    msg = build_adjudicator_user_message("output", votes)
    assert "compliance" in msg
    assert "refusal" in msg
    assert "clarification" in msg


def test_adjudicator_message_has_no_forbidden_fields():
    votes = [_make_vote("compliance")] * 3
    msg = build_adjudicator_user_message("output text", votes)
    assert_no_forbidden_fields(msg)


def test_adjudicator_message_contains_confidences():
    votes = [_make_vote("compliance", 0.95)]
    msg = build_adjudicator_user_message("output", votes)
    assert "0.95" in msg


# ── load_default_prompts ─────────────────────────────────────────────────────

def test_load_default_prompts_has_all_keys():
    prompts = load_default_prompts()
    for key in ("A", "B", "C", "adjudicator"):
        assert key in prompts, f"Missing key: {key}"


def test_load_default_prompts_nonempty():
    prompts = load_default_prompts()
    for key, text in prompts.items():
        assert len(text) > 50, f"Prompt {key!r} seems too short"


def test_load_default_prompts_contain_label_names():
    """Each judge prompt must mention the 6 label names."""
    labels = ["compliance", "refusal", "clarification",
              "hierarchy_preservation", "source_isolation", "safe_redirection"]
    prompts = load_default_prompts()
    for key in ("A", "B", "C"):
        text = prompts[key]
        for label in labels:
            assert label in text, f"Prompt {key!r} missing label {label!r}"
