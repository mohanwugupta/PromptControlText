"""
mining/features.py
==================
Milestone 2 — Deterministic feature extraction (PRD v3 §10).

Features are organised into four groups:

* **length**        — character count, word count, sentence count
* **punctuation**   — question-mark count, exclamation-mark count
* **lexical**       — phrase-group hit counts (refusal, hedging, apology,
                      redirection)
* **language**      — ISO 639-1 language code + a binary ``is_english``
                      flag (requires *langdetect*)

All features are deterministic: identical input → identical output.
Missing / empty text returns a zero-vector with ``is_english=False`` and
``language="unknown"``.

The ``extract_features`` function returns a *dict* so callers can choose
whether to materialise it as a new column, explode it into many columns,
or feed it directly to sklearn.
"""

from __future__ import annotations

import re
from typing import Dict

import pandas as pd

# ---------------------------------------------------------------------------
# Phrase dictionaries
# ---------------------------------------------------------------------------
_REFUSAL_PHRASES = [
    r"i cannot\b", r"i can'?t\b", r"i will not\b", r"i won'?t\b",
    r"i'?m unable to", r"i'?m not able to", r"i must decline",
    r"i should not\b", r"i apologize,? but", r"as an ai",
    r"i am programmed to be", r"i cannot fulfill",
    r"it is not (appropriate|ethical|legal|right)\b",
    r"i (do not|don't) (feel comfortable|think (this|it) is)",
    r"sorry,? (but\s+)?(i\s+)?(can'?t|cannot|won'?t|am unable)",
    r"not (something|a topic|a request) (i|that i) (can|should|am able to|will)",
    r"(against|violates?) (my (programming|guidelines|values)|ethical)",
]

_HEDGING_PHRASES = [
    r"\bi think\b", r"\bi believe\b", r"\bperhaps\b", r"\bmaybe\b",
    r"\bit'?s possible\b", r"\bit seems\b", r"\bappears to\b",
    r"\bmight\b", r"\bcould be\b", r"\bone could argue\b",
    r"\bgenerally speaking\b", r"\btypically\b", r"\bin general\b",
    r"\bsome might\b", r"\bit depends\b",
]

_APOLOGY_PHRASES = [
    r"\bi'?m sorry\b", r"\bi apologize\b", r"\bmy apologies\b",
    r"\bi regret\b", r"\bunfortunately\b", r"\bi'?m afraid\b",
    r"\bregrettably\b",
]

_REDIRECTION_PHRASES = [
    r"\bhowever,?\s+i (can|could|would)\b",
    r"\binstead,?\s+i (can|could|would|suggest)\b",
    r"\bi can (suggest|recommend|offer|provide)\b",
    r"\bconsider (talking|speaking|reaching|contacting)\b",
    r"\bplease (consult|see|refer|check)\b",
    r"\bi'?d (suggest|recommend)\b",
    r"\ban alternative\b",
    r"\ba (better|safer|more appropriate) (way|approach|option)\b",
]

_QUESTION_PATTERNS = [r"\?"]

_COMPILED: dict[str, list] = {}


def _compile_group(name: str, patterns: list[str]) -> list:
    if name not in _COMPILED:
        _COMPILED[name] = [re.compile(p, re.IGNORECASE) for p in patterns]
    return _COMPILED[name]


def _count_matches(text_lower: str, patterns: list) -> int:
    return sum(1 for p in patterns if p.search(text_lower))


# ---------------------------------------------------------------------------
# Language detection (optional dependency)
# ---------------------------------------------------------------------------
def _detect_language(text: str) -> str:
    """Return ISO 639-1 code or 'unknown' on failure."""
    try:
        from langdetect import detect, LangDetectException  # type: ignore
        return detect(text)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_features(text: str) -> Dict[str, object]:
    """
    Extract a flat feature dict from a single response string.

    Parameters
    ----------
    text : str
        Raw model output text.

    Returns
    -------
    dict with keys:
        char_count, word_count, sentence_count,
        question_count, exclamation_count,
        refusal_phrase_count, hedging_phrase_count,
        apology_phrase_count, redirection_phrase_count,
        language, is_english
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "question_count": 0,
            "exclamation_count": 0,
            "refusal_phrase_count": 0,
            "hedging_phrase_count": 0,
            "apology_phrase_count": 0,
            "redirection_phrase_count": 0,
            "language": "unknown",
            "is_english": False,
        }

    tl = text.lower()

    # Length
    char_count = len(text)
    word_count = len(text.split())
    # Sentences: split on .  !  ?  followed by whitespace or end
    sentence_count = max(1, len(re.split(r"[.!?]+\s+|[.!?]+$", text.strip())))

    # Punctuation
    question_count = text.count("?")
    exclamation_count = text.count("!")

    # Phrase groups
    refusal_count = _count_matches(
        tl, _compile_group("refusal", _REFUSAL_PHRASES)
    )
    hedging_count = _count_matches(
        tl, _compile_group("hedging", _HEDGING_PHRASES)
    )
    apology_count = _count_matches(
        tl, _compile_group("apology", _APOLOGY_PHRASES)
    )
    redirection_count = _count_matches(
        tl, _compile_group("redirection", _REDIRECTION_PHRASES)
    )

    # Language
    lang = _detect_language(text)
    is_english = lang == "en"

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "refusal_phrase_count": refusal_count,
        "hedging_phrase_count": hedging_count,
        "apology_phrase_count": apology_count,
        "redirection_phrase_count": redirection_count,
        "language": lang,
        "is_english": is_english,
    }


FEATURE_NUMERIC_COLS = [
    "char_count",
    "word_count",
    "sentence_count",
    "question_count",
    "exclamation_count",
    "refusal_phrase_count",
    "hedging_phrase_count",
    "apology_phrase_count",
    "redirection_phrase_count",
]

FEATURE_ALL_COLS = FEATURE_NUMERIC_COLS + ["language", "is_english"]


def extract_features_df(df: pd.DataFrame, text_col: str = "model_output") -> pd.DataFrame:
    """
    Apply ``extract_features`` to every row of *df* and return a new
    DataFrame (same index) with all feature columns appended.

    The original columns of *df* are preserved; feature columns are added.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    feature_records = df[text_col].apply(lambda t: extract_features(str(t) if pd.notna(t) else ""))
    feat_df = pd.DataFrame(feature_records.tolist(), index=df.index)
    return pd.concat([df, feat_df], axis=1)
