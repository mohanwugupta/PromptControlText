"""tests/test_feature_extraction.py — Milestone 2 tests."""

import pandas as pd
import pytest

from mining.features import (
    extract_features,
    extract_features_df,
    FEATURE_NUMERIC_COLS,
    FEATURE_ALL_COLS,
)


# ── determinism ─────────────────────────────────────────────────────────────

def test_deterministic_on_fixed_input():
    text = "I cannot help with that. I'm sorry, but I can't assist."
    r1 = extract_features(text)
    r2 = extract_features(text)
    assert r1 == r2


# ── missing / empty text ────────────────────────────────────────────────────

def test_empty_string_returns_zero_vector():
    r = extract_features("")
    for col in FEATURE_NUMERIC_COLS:
        assert r[col] == 0, f"{col} should be 0 for empty string"
    assert r["language"] == "unknown"
    assert r["is_english"] is False


def test_none_like_empty_string():
    r = extract_features(None)
    for col in FEATURE_NUMERIC_COLS:
        assert r[col] == 0


def test_whitespace_only_returns_zeros():
    r = extract_features("   \n\t  ")
    assert r["char_count"] == 0


# ── length features ─────────────────────────────────────────────────────────

def test_char_count_correct():
    text = "hello world"
    r = extract_features(text)
    assert r["char_count"] == len(text)


def test_word_count_correct():
    text = "one two three four"
    r = extract_features(text)
    assert r["word_count"] == 4


def test_sentence_count_at_least_one():
    r = extract_features("No periods here")
    assert r["sentence_count"] >= 1


def test_sentence_count_multiple():
    r = extract_features("First sentence. Second sentence. Third one!")
    assert r["sentence_count"] >= 2


# ── punctuation features ─────────────────────────────────────────────────────

def test_question_count():
    r = extract_features("Are you sure? What do you mean? Really?")
    assert r["question_count"] == 3


def test_exclamation_count():
    r = extract_features("Yes! No! Maybe!")
    assert r["exclamation_count"] == 3


# ── lexical phrase groups ────────────────────────────────────────────────────

def test_refusal_phrase_detected():
    r = extract_features("I cannot help you with this request.")
    assert r["refusal_phrase_count"] >= 1


def test_no_refusal_in_compliance_text():
    r = extract_features("Sure, here is how to bake a cake. Step 1: preheat oven.")
    assert r["refusal_phrase_count"] == 0


def test_hedging_phrase_detected():
    r = extract_features("I think this might be the right approach, perhaps.")
    assert r["hedging_phrase_count"] >= 1


def test_apology_phrase_detected():
    r = extract_features("I'm sorry, I apologize for the confusion.")
    assert r["apology_phrase_count"] >= 1


def test_redirection_phrase_detected():
    r = extract_features("Instead, I would recommend speaking with an expert.")
    assert r["redirection_phrase_count"] >= 1


def test_multiple_groups_can_co_occur():
    text = "I'm sorry but I cannot help. Perhaps you should consult a professional instead."
    r = extract_features(text)
    assert r["refusal_phrase_count"] >= 1
    assert r["apology_phrase_count"] >= 1


# ── feature names stability ─────────────────────────────────────────────────

def test_feature_keys_stable():
    r = extract_features("some text")
    assert set(r.keys()) == set(FEATURE_ALL_COLS)


# ── DataFrame helper ────────────────────────────────────────────────────────

def test_extract_features_df_appends_columns():
    df = pd.DataFrame({"model_output": ["I cannot help.", "Sure, here is how.", ""]})
    result = extract_features_df(df)
    for col in FEATURE_ALL_COLS:
        assert col in result.columns


def test_extract_features_df_preserves_original_columns():
    df = pd.DataFrame({"model_output": ["hello"], "other": [42]})
    result = extract_features_df(df)
    assert "other" in result.columns
    assert result["other"].iloc[0] == 42


def test_extract_features_df_missing_col_raises():
    df = pd.DataFrame({"text": ["hello"]})
    with pytest.raises(ValueError, match="model_output"):
        extract_features_df(df)


def test_extract_features_df_row_count_preserved():
    df = pd.DataFrame({"model_output": ["a", "b", "c"]})
    result = extract_features_df(df)
    assert len(result) == 3
