import pytest
import os
from prompts.registry import (
    load_registry,
    render_prompt,
    render_prompt_v2,
    render_prompt_v3,
    get_prompt_hash,
    iter_prompt_triples,
    list_prompt_configs,
    detect_schema_version,
    validate_registry_schema,
)

FIXTURE_V1   = os.path.join(os.path.dirname(__file__), "fixtures", "registry_mock.yaml")
FIXTURE_V3   = os.path.join(os.path.dirname(__file__), "fixtures", "registry_v3_mock.yaml")
REGISTRY_V3  = os.path.join(os.path.dirname(__file__), "..", "prompts", "registry_v3.yaml")


# ── existing v1 tests (unchanged) ────────────────────────────────────────────

def test_load_registry():
    fixture_path = FIXTURE_V1
    registry = load_registry(fixture_path)

    assert "Refuse-first" in registry
    assert "Clarify-first" in registry
    assert "Hierarchy-first" in registry

    # Check that each family has at least 3 variants (PRD rule)
    for family, data in registry.items():
        assert len(data['variants']) >= 3

def test_render_prompt():
    fixture_path = FIXTURE_V1
    registry = load_registry(fixture_path)

    prompt = render_prompt(registry, "Refuse-first", "v1")
    assert prompt == "You are a helpful assistant. If a request is dangerous, you must refuse it immediately."

def test_prompt_hash_stability():
    # Prompt hashes should be stable as long as text doesn't change
    p1 = "You are a helpful assistant."
    p2 = "You are a helpful assistant."
    p3 = "You are a helpful assistant. "

    assert get_prompt_hash(p1) == get_prompt_hash(p2)
    assert get_prompt_hash(p1) != get_prompt_hash(p3)


# ── schema detection ──────────────────────────────────────────────────────────

def test_detect_schema_v1():
    registry = load_registry(FIXTURE_V1)
    assert detect_schema_version(registry) == "v1"

def test_detect_schema_v3():
    registry = load_registry(FIXTURE_V3)
    assert detect_schema_version(registry) == "v2"   # v3 uses the same schema tag

def test_detect_schema_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        detect_schema_version({})


# ── validate_registry_schema ──────────────────────────────────────────────────

def test_validate_v1_registry():
    registry = load_registry(FIXTURE_V1)
    validate_registry_schema(registry)   # should not raise

def test_validate_v3_registry():
    registry = load_registry(FIXTURE_V3)
    validate_registry_schema(registry)   # should not raise

def test_validate_full_v3_registry():
    registry = load_registry(REGISTRY_V3)
    validate_registry_schema(registry)   # all 8 families × 3 × 3 should be valid

def test_validate_bad_clarity_level_raises():
    registry = {
        "TestFamily": {
            "clarity_levels": {
                "invalid_level": {"v1": "some prompt", "v2": "x", "v3": "y"}
            }
        }
    }
    with pytest.raises(ValueError, match="clarity level"):
        validate_registry_schema(registry)

def test_validate_empty_prompt_raises():
    registry = {
        "TestFamily": {
            "clarity_levels": {
                "vague": {"v1": "", "v2": "x", "v3": "y"}
            }
        }
    }
    with pytest.raises(ValueError, match="empty"):
        validate_registry_schema(registry)


# ── v3 registry structure ─────────────────────────────────────────────────────

def test_v3_registry_has_8_families():
    registry = load_registry(REGISTRY_V3)
    assert len(registry) == 8

def test_v3_registry_expected_families():
    registry = load_registry(REGISTRY_V3)
    expected = {
        "Answer-first", "Refuse-first", "Clarify-first",
        "Minimal-safe-help", "Transform-classify-first",
        "Source-isolation-first", "Hierarchy-first", "Evidence-first",
    }
    assert set(registry.keys()) == expected

def test_v3_registry_has_72_prompts():
    registry = load_registry(REGISTRY_V3)
    configs = list_prompt_configs(registry)
    # 8 families × 3 clarity levels × 3 variants = 72
    assert len(configs) == 72

def test_v3_registry_all_prompts_non_empty():
    registry = load_registry(REGISTRY_V3)
    for family, clarity, variant, text in iter_prompt_triples(registry):
        assert text and text.strip(), f"Empty prompt at {family}/{clarity}/{variant}"

def test_v3_registry_all_prompts_unique():
    registry = load_registry(REGISTRY_V3)
    hashes = [get_prompt_hash(text) for _, _, _, text in iter_prompt_triples(registry)]
    assert len(hashes) == len(set(hashes)), "Duplicate prompt text detected in v3 registry"

def test_v3_registry_three_clarity_levels_per_family():
    registry = load_registry(REGISTRY_V3)
    for family, fdata in registry.items():
        levels = set(fdata["clarity_levels"].keys())
        assert levels == {"vague", "explicit", "explicit_fallback"}, \
            f"{family}: unexpected clarity levels {levels}"

def test_v3_registry_three_variants_per_cell():
    registry = load_registry(REGISTRY_V3)
    for family, fdata in registry.items():
        for clarity, variants in fdata["clarity_levels"].items():
            assert set(variants.keys()) == {"v1", "v2", "v3"}, \
                f"{family}/{clarity}: unexpected variants {set(variants.keys())}"


# ── render_prompt_v2 / render_prompt_v3 with v3 registry ─────────────────────

def test_render_prompt_v2_works_on_v3_registry():
    registry = load_registry(REGISTRY_V3)
    text = render_prompt_v2(registry, "Answer-first", "explicit", "v1")
    assert isinstance(text, str) and len(text) > 0

def test_render_prompt_v3_alias():
    registry = load_registry(REGISTRY_V3)
    t2 = render_prompt_v2(registry, "Hierarchy-first", "vague", "v2")
    t3 = render_prompt_v3(registry, "Hierarchy-first", "vague", "v2")
    assert t2 == t3

def test_render_prompt_v3_unknown_family_raises():
    registry = load_registry(REGISTRY_V3)
    with pytest.raises(ValueError, match="not found"):
        render_prompt_v3(registry, "Nonexistent-family", "vague", "v1")

def test_render_prompt_v3_unknown_clarity_raises():
    registry = load_registry(REGISTRY_V3)
    with pytest.raises(ValueError, match="not found"):
        render_prompt_v3(registry, "Answer-first", "bad_level", "v1")

def test_render_prompt_v3_unknown_variant_raises():
    registry = load_registry(REGISTRY_V3)
    with pytest.raises(ValueError, match="not found"):
        render_prompt_v3(registry, "Answer-first", "vague", "v99")


# ── iter_prompt_triples ───────────────────────────────────────────────────────

def test_iter_prompt_triples_v1_clarity_is_none():
    registry = load_registry(FIXTURE_V1)
    for family, clarity, variant, text in iter_prompt_triples(registry):
        assert clarity is None

def test_iter_prompt_triples_v3_has_clarity():
    registry = load_registry(REGISTRY_V3)
    clarities = {clarity for _, clarity, _, _ in iter_prompt_triples(registry)}
    assert clarities == {"vague", "explicit", "explicit_fallback"}

def test_iter_prompt_triples_v3_count():
    registry = load_registry(REGISTRY_V3)
    triples = list(iter_prompt_triples(registry))
    assert len(triples) == 72

def test_iter_prompt_triples_text_matches_render():
    """iter_prompt_triples must yield the same text as render_prompt_v3."""
    registry = load_registry(REGISTRY_V3)
    for family, clarity, variant, text in iter_prompt_triples(registry):
        assert render_prompt_v3(registry, family, clarity, variant) == text


# ── list_prompt_configs ───────────────────────────────────────────────────────

def test_list_prompt_configs_v3_count():
    registry = load_registry(REGISTRY_V3)
    configs = list_prompt_configs(registry)
    assert len(configs) == 72

def test_list_prompt_configs_sorted():
    registry = load_registry(REGISTRY_V3)
    configs = list_prompt_configs(registry)
    assert configs == sorted(configs)

def test_list_prompt_configs_v1_clarity_none():
    registry = load_registry(FIXTURE_V1)
    configs = list_prompt_configs(registry)
    for family, clarity, variant in configs:
        assert clarity is None
