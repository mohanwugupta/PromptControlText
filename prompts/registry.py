import yaml
import hashlib
from typing import Dict, Any

def load_registry(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        registry = yaml.safe_load(f)
    return registry

def render_prompt(registry: Dict[str, Any], family: str, variant: str) -> str:
    if family not in registry:
        raise ValueError(f"Prompt family {family} not found in registry.")
    if variant not in registry[family]['variants']:
        raise ValueError(f"Variant {variant} not found in family {family}.")
    return registry[family]['variants'][variant]

def get_prompt_hash(prompt_text: str) -> str:
    # Stable hash of the prompt text to lock prompt versions
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()


def render_prompt_v2(registry: Dict[str, Any], family: str, clarity: str, variant: str) -> str:
    """Render a v2 prompt: family × clarity_level × paraphrase variant."""
    if family not in registry:
        raise ValueError(f"Prompt family '{family}' not found in registry.")
    clarity_levels = registry[family].get("clarity_levels", {})
    if clarity not in clarity_levels:
        raise ValueError(f"Clarity level '{clarity}' not found in family '{family}'.")
    variants = clarity_levels[clarity]
    if variant not in variants:
        raise ValueError(f"Variant '{variant}' not found in {family}/{clarity}.")
    return variants[variant]
