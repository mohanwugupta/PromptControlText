from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any

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

    model_config = ConfigDict(extra='forbid')
