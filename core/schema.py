from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Dict, Any, Literal

# PRD v2 §10-§12: Allowed values for policy-routing annotations
POLICY_LABELS = ("answer", "refuse", "clarify", "minimal_safe_help", "hierarchy_preserve")
AMBIGUITY_LEVELS = ("low", "medium", "high")
CONTEXT_CONDITIONS = ("clean", "ambiguous", "quoted", "conflict", "override", "attacked")
CLARITY_LEVELS = ("vague", "explicit", "explicit_fallback")


class EvalItem(BaseModel):
    item_id: str
    benchmark: str
    domain: str
    input_text: str
    gold_label: str
    prompt_family: Optional[str] = None
    prompt_variant: Optional[str] = None
    model_output: Optional[str] = None
    score: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    # PRD v2 fields: policy-routing annotations
    policy_label: Optional[str] = None
    ambiguity_level: Optional[str] = None
    context_condition: Optional[str] = None
    clarity_level: Optional[str] = None

    model_config = ConfigDict(extra='forbid')

    @field_validator("policy_label")
    @classmethod
    def _validate_policy_label(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in POLICY_LABELS:
            raise ValueError(f"policy_label must be one of {POLICY_LABELS}, got '{v}'")
        return v

    @field_validator("ambiguity_level")
    @classmethod
    def _validate_ambiguity_level(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in AMBIGUITY_LEVELS:
            raise ValueError(f"ambiguity_level must be one of {AMBIGUITY_LEVELS}, got '{v}'")
        return v

    @field_validator("context_condition")
    @classmethod
    def _validate_context_condition(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in CONTEXT_CONDITIONS:
            raise ValueError(f"context_condition must be one of {CONTEXT_CONDITIONS}, got '{v}'")
        return v

    @field_validator("clarity_level")
    @classmethod
    def _validate_clarity_level(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in CLARITY_LEVELS:
            raise ValueError(f"clarity_level must be one of {CLARITY_LEVELS}, got '{v}'")
        return v
