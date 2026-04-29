import yaml
import hashlib
from typing import Dict, Any, Iterator, Tuple, List


def load_registry(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        registry = yaml.safe_load(f)
    return registry


def render_prompt(registry: Dict[str, Any], family: str, variant: str) -> str:
    """Render a v1 prompt: family × variant."""
    if family not in registry:
        raise ValueError(f"Prompt family {family} not found in registry.")
    if variant not in registry[family]['variants']:
        raise ValueError(f"Variant {variant} not found in family {family}.")
    return registry[family]['variants'][variant]


def get_prompt_hash(prompt_text: str) -> str:
    # Stable hash of the prompt text to lock prompt versions
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()


def render_prompt_v2(registry: Dict[str, Any], family: str, clarity: str, variant: str) -> str:
    """Render a v2/v3 prompt: family × clarity_level × paraphrase variant."""
    if family not in registry:
        raise ValueError(f"Prompt family '{family}' not found in registry.")
    clarity_levels = registry[family].get("clarity_levels", {})
    if clarity not in clarity_levels:
        raise ValueError(f"Clarity level '{clarity}' not found in family '{family}'.")
    variants = clarity_levels[clarity]
    if variant not in variants:
        raise ValueError(f"Variant '{variant}' not found in {family}/{clarity}.")
    return variants[variant]


# ---------------------------------------------------------------------------
# render_prompt_v3 is an alias for render_prompt_v2.
# The v3 YAML uses the identical schema (family → clarity_levels → variants).
# The alias is provided so callers can be explicit about which registry they
# are using without having to know that the underlying format is the same.
# ---------------------------------------------------------------------------
render_prompt_v3 = render_prompt_v2


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

def detect_schema_version(registry: Dict[str, Any]) -> str:
    """
    Infer whether *registry* uses the v1 (variants) or v2/v3 (clarity_levels)
    schema by inspecting the first family entry.

    Returns
    -------
    "v1"  if the registry uses ``family → variants``
    "v2"  if the registry uses ``family → clarity_levels``
    """
    if not registry:
        raise ValueError("Registry is empty.")
    first = next(iter(registry.values()))
    if "clarity_levels" in first:
        return "v2"
    if "variants" in first:
        return "v1"
    raise ValueError(
        "Cannot detect schema version: first family has neither "
        "'clarity_levels' nor 'variants' key."
    )


# ---------------------------------------------------------------------------
# Iteration helpers — work for both v1 and v2/v3 schemas
# ---------------------------------------------------------------------------

def iter_prompt_triples(
    registry: Dict[str, Any],
) -> Iterator[Tuple[str, str, str, str]]:
    """
    Yield every (family, clarity_or_none, variant, prompt_text) tuple in
    *registry*, regardless of schema version.

    For **v1** registries the yielded ``clarity_or_none`` is ``None``.
    For **v2/v3** registries it is the clarity level string
    (``"vague"``, ``"explicit"``, or ``"explicit_fallback"``).

    This is the canonical iteration interface for experiment runners so they
    do not need to branch on schema version themselves.
    """
    schema = detect_schema_version(registry)
    if schema == "v1":
        for family, fdata in registry.items():
            for variant, text in fdata["variants"].items():
                yield family, None, variant, text
    else:  # v2 / v3
        for family, fdata in registry.items():
            for clarity, variants in fdata.get("clarity_levels", {}).items():
                for variant, text in variants.items():
                    yield family, clarity, variant, text


def list_prompt_configs(
    registry: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    """
    Return a sorted list of (family, clarity_or_none, variant) tuples.

    Useful for display, validation, and counting (len == number of prompts).
    """
    return sorted(
        (family, clarity, variant)
        for family, clarity, variant, _ in iter_prompt_triples(registry)
    )


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

_EXPECTED_CLARITY_LEVELS = {"vague", "explicit", "explicit_fallback"}
_EXPECTED_VARIANTS = {"v1", "v2", "v3"}


def validate_registry_schema(registry: Dict[str, Any]) -> None:
    """
    Validate that *registry* conforms to its detected schema.

    Raises
    ------
    ValueError  with a descriptive message on the first detected violation.
    """
    if not registry:
        raise ValueError("Registry is empty.")

    schema = detect_schema_version(registry)

    if schema == "v1":
        for family, fdata in registry.items():
            if "variants" not in fdata:
                raise ValueError(f"[{family}] missing 'variants' key.")
            for variant in fdata["variants"]:
                if not isinstance(fdata["variants"][variant], str):
                    raise ValueError(
                        f"[{family}/{variant}] prompt text must be a string."
                    )
    else:
        for family, fdata in registry.items():
            if "clarity_levels" not in fdata:
                raise ValueError(f"[{family}] missing 'clarity_levels' key.")
            for clarity, variants in fdata["clarity_levels"].items():
                if clarity not in _EXPECTED_CLARITY_LEVELS:
                    raise ValueError(
                        f"[{family}] unexpected clarity level '{clarity}'. "
                        f"Expected one of {_EXPECTED_CLARITY_LEVELS}."
                    )
                for variant, text in variants.items():
                    if variant not in _EXPECTED_VARIANTS:
                        raise ValueError(
                            f"[{family}/{clarity}] unexpected variant '{variant}'."
                        )
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(
                            f"[{family}/{clarity}/{variant}] prompt text is empty."
                        )

