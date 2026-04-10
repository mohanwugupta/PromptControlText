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
