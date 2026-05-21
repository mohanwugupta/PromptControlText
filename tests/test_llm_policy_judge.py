"""
tests/test_llm_policy_judge.py

TDD tests for scoring/llm_policy_judge.py
RED → GREEN
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from scoring.llm_policy_judge import judge_row, JUDGE_IDS, _row_hash


# ── helpers ───────────────────────────────────────────────────────────────────

_VALID_RECORD = {
    "primary_label": "compliance",
    "secondary_label": None,
    "confidence": 0.9,
    "contains_answer": True,
    "contains_refusal": False,
    "contains_clarifying_question": False,
    "contains_safe_redirect": False,
    "mentions_instruction_priority": False,
    "treats_external_text_as_data": False,
    "evidence": "answered directly",
    "reason": "full response",
}


def _make_client(record: Optional[Dict] = None, error: bool = False):
    """Return a mock VLLMClient whose generate() returns valid JSON."""
    client = MagicMock()
    if error:
        client.generate.side_effect = Exception("network error")
    else:
        raw = json.dumps(record or _VALID_RECORD)
        client.generate.return_value = (raw, None)
    return client


def _default_prompts():
    return {"A": "You are judge A.", "B": "You are judge B.", "C": "You are judge C."}


# ── _row_hash ─────────────────────────────────────────────────────────────────

def test_row_hash_is_deterministic():
    assert _row_hash("hello") == _row_hash("hello")


def test_row_hash_differs_for_different_inputs():
    assert _row_hash("hello") != _row_hash("world")


def test_row_hash_is_sha256():
    expected = hashlib.sha256("test".encode()).hexdigest()
    assert _row_hash("test") == expected


# ── judge_row: happy path ─────────────────────────────────────────────────────

def test_judge_row_returns_three_votes():
    client = _make_client()
    votes = judge_row(
        row_index=0,
        job_id="test_job",
        model_output="The answer is 42.",
        prompts=_default_prompts(),
        client=client,
        model="test-model",
    )
    assert len(votes) == 3


def test_judge_row_each_vote_has_required_keys():
    client = _make_client()
    votes = judge_row(
        row_index=0,
        job_id="job1",
        model_output="output",
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    required = {"job_id", "row_index", "row_hash", "judge_id",
                "judge_prompt_variant", "judge_model", "parse_error",
                "primary_label", "confidence"}
    for vote in votes:
        assert required <= vote.keys(), f"Missing keys: {required - vote.keys()}"


def test_judge_row_vote_judge_ids_are_A_B_C():
    client = _make_client()
    votes = judge_row(
        row_index=5,
        job_id="j",
        model_output="x",
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    assert [v["judge_id"] for v in votes] == list(JUDGE_IDS)


def test_judge_row_row_hash_consistent():
    client = _make_client()
    output = "consistent output"
    votes = judge_row(
        row_index=0,
        job_id="j",
        model_output=output,
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    expected_hash = _row_hash(output)
    for vote in votes:
        assert vote["row_hash"] == expected_hash


def test_judge_row_job_id_propagated():
    client = _make_client()
    votes = judge_row(
        row_index=0,
        job_id="my_job",
        model_output="x",
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    for vote in votes:
        assert vote["job_id"] == "my_job"


def test_judge_row_successful_vote_has_no_parse_error():
    client = _make_client()
    votes = judge_row(
        row_index=0,
        job_id="j",
        model_output="x",
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    for vote in votes:
        assert vote["parse_error"] == ""


def test_judge_row_sets_primary_label_from_record():
    client = _make_client({**_VALID_RECORD, "primary_label": "refusal"})
    votes = judge_row(
        row_index=0,
        job_id="j",
        model_output="x",
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    for vote in votes:
        assert vote["primary_label"] == "refusal"


# ── judge_row: error handling ─────────────────────────────────────────────────

def test_judge_row_handles_client_exception():
    client = _make_client(error=True)
    votes = judge_row(
        row_index=0,
        job_id="j",
        model_output="x",
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    # All three should be parse_error sentinels
    assert len(votes) == 3
    for vote in votes:
        assert vote["primary_label"] == "parse_error"
        assert vote["parse_error"] != ""


def test_judge_row_handles_invalid_json():
    client = MagicMock()
    client.generate.return_value = ("not json {{{", None)
    votes = judge_row(
        row_index=0,
        job_id="j",
        model_output="x",
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    for vote in votes:
        assert vote["primary_label"] == "parse_error"


def test_judge_row_retries_on_first_failure():
    """Client fails once then succeeds — vote should be populated."""
    client = MagicMock()
    raw = json.dumps(_VALID_RECORD)
    # First call: error; second call (retry): success
    client.generate.side_effect = [Exception("fail"), (raw, None)]
    votes = judge_row(
        row_index=0,
        job_id="j",
        model_output="x",
        prompts={"A": "p", "B": "p", "C": "p"},
        client=client,
        model="m",
    )
    # Judge A should have recovered on retry
    assert votes[0]["primary_label"] == "compliance"


def test_judge_row_sentinel_vote_is_structurally_complete():
    """Even on total failure, each vote must have all expected keys."""
    client = _make_client(error=True)
    votes = judge_row(
        row_index=0,
        job_id="j",
        model_output="x",
        prompts=_default_prompts(),
        client=client,
        model="m",
    )
    bool_fields = [
        "contains_answer", "contains_refusal", "contains_clarifying_question",
        "contains_safe_redirect", "mentions_instruction_priority",
        "treats_external_text_as_data",
    ]
    for vote in votes:
        for f in bool_fields:
            assert f in vote


def test_judge_row_model_name_propagated():
    client = _make_client()
    votes = judge_row(
        row_index=0,
        job_id="j",
        model_output="x",
        prompts=_default_prompts(),
        client=client,
        model="Qwen2.5-72B",
    )
    for vote in votes:
        assert vote["judge_model"] == "Qwen2.5-72B"
