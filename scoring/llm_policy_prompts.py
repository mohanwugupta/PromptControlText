"""
scoring/llm_policy_prompts.py

Build output-only judge inputs and load prompt templates.
Enforces that no forbidden metadata enters the judge input.
"""
from __future__ import annotations

import pathlib
from typing import Dict, List

# Fields that must NEVER appear in the judge input
_FORBIDDEN_FIELDS = frozenset([
    "input_text", "benchmark", "prompt_family", "clarity_level",
    "gold_label", "policy_label", "prompt_variant", "ambiguity_level",
    "context_condition", "model_name", "item_id", "domain",
    "classified_policy",  # old rule-based label
])

_PROMPT_DIR = pathlib.Path(__file__).parent


def load_prompt(path: str | pathlib.Path) -> str:
    with open(path) as f:
        return f.read().strip()


def load_default_prompts() -> Dict[str, str]:
    """Return the three judge prompts and adjudicator prompt keyed by role."""
    return {
        "A": load_prompt(_PROMPT_DIR / "llm_policy_judge_prompt_A_v1.txt"),
        "B": load_prompt(_PROMPT_DIR / "llm_policy_judge_prompt_B_v1.txt"),
        "C": load_prompt(_PROMPT_DIR / "llm_policy_judge_prompt_C_v1.txt"),
        "adjudicator": load_prompt(_PROMPT_DIR / "llm_policy_adjudicator_prompt_v1.txt"),
    }


def get_model_output(row: Dict) -> str:
    """
    Extract the model output from a CSV row using the canonical column name
    or the accepted alias.  Raises ValueError on missing/empty output.
    """
    if "model_output" in row and "model_out" in row:
        if row["model_output"] != row["model_out"]:
            import logging
            logging.getLogger(__name__).warning(
                "Row has both model_output and model_out with different values; "
                "using model_output."
            )
    text = row.get("model_output") or row.get("model_out")
    if not text or str(text).strip() == "" or str(text).lower() in ("nan", "none"):
        raise ValueError("Row has no non-empty model_output / model_out value.")
    return str(text).strip()


def build_judge_user_message(model_output: str) -> str:
    """
    Build the user-turn content sent to a first-pass judge.
    Contains ONLY the assistant response — no metadata whatsoever.
    """
    return f"assistant_response:\n{model_output}"


def build_adjudicator_user_message(
    model_output: str,
    votes: List[Dict],
) -> str:
    """
    Build the user-turn content sent to an adjudicator.
    Includes the model output, the three first-pass labels, confidences,
    and evidence strings — nothing else.
    """
    lines = [f"assistant_response:\n{model_output}\n"]
    for i, v in enumerate(votes, 1):
        lines.append(
            f"judge_{i}: label={v['primary_label']}  "
            f"confidence={v['confidence']:.2f}  "
            f"evidence={v.get('evidence', '')}"
        )
    return "\n".join(lines)


def assert_no_forbidden_fields(user_message: str) -> None:
    """
    Raise AssertionError if any forbidden metadata field name appears
    verbatim in the judge user message.
    """
    lower = user_message.lower()
    for field in _FORBIDDEN_FIELDS:
        assert field.lower() not in lower, (
            f"Forbidden field '{field}' found in judge user message."
        )
