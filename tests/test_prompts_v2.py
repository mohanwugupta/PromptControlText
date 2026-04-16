"""
RED tests for PRD v2 prompt registry.

PRD v2 §9: 4 controller families × 3 clarity levels × 3 paraphrases = 36 prompts.
Scientific rule: claims are over family aggregates, not single prompts.
"""
import pytest
import os
from prompts.registry import load_registry, render_prompt_v2, get_prompt_hash

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
V2_REGISTRY = os.path.join(FIXTURE_DIR, "registry_v2_mock.yaml")

REQUIRED_FAMILIES = ["Refuse-first", "Clarify-first", "Minimal-safe-help", "Hierarchy-first"]
REQUIRED_CLARITY = ["vague", "explicit", "explicit_fallback"]


def test_v2_registry_loads():
    registry = load_registry(V2_REGISTRY)
    assert isinstance(registry, dict)


def test_v2_registry_has_all_four_families():
    registry = load_registry(V2_REGISTRY)
    for family in REQUIRED_FAMILIES:
        assert family in registry, f"Missing controller family: {family}"


def test_v2_registry_families_have_clarity_levels():
    registry = load_registry(V2_REGISTRY)
    for family in REQUIRED_FAMILIES:
        assert "clarity_levels" in registry[family], f"{family} missing clarity_levels"
        for cl in REQUIRED_CLARITY:
            assert cl in registry[family]["clarity_levels"], (
                f"{family} missing clarity level: {cl}"
            )


def test_v2_registry_each_cell_has_three_paraphrases():
    """PRD v2 scientific rule: >=3 paraphrases per family × clarity cell."""
    registry = load_registry(V2_REGISTRY)
    for family in REQUIRED_FAMILIES:
        for cl in REQUIRED_CLARITY:
            variants = registry[family]["clarity_levels"][cl]
            assert len(variants) >= 3, (
                f"{family}/{cl} has only {len(variants)} paraphrases (need >=3)"
            )


def test_v2_render_prompt():
    registry = load_registry(V2_REGISTRY)
    prompt = render_prompt_v2(registry, "Refuse-first", "vague", "v1")
    assert isinstance(prompt, str)
    assert len(prompt) > 10


def test_v2_render_prompt_different_clarity_different_text():
    """Different clarity levels must produce different prompt text."""
    registry = load_registry(V2_REGISTRY)
    vague = render_prompt_v2(registry, "Refuse-first", "vague", "v1")
    explicit = render_prompt_v2(registry, "Refuse-first", "explicit", "v1")
    explicit_fb = render_prompt_v2(registry, "Refuse-first", "explicit_fallback", "v1")
    assert vague != explicit
    assert explicit != explicit_fb
    assert vague != explicit_fb


def test_v2_render_prompt_invalid_family_raises():
    registry = load_registry(V2_REGISTRY)
    with pytest.raises(ValueError, match="family"):
        render_prompt_v2(registry, "NonexistentFamily", "vague", "v1")


def test_v2_render_prompt_invalid_clarity_raises():
    registry = load_registry(V2_REGISTRY)
    with pytest.raises(ValueError, match="clarity"):
        render_prompt_v2(registry, "Refuse-first", "nonexistent_clarity", "v1")


def test_v2_render_prompt_invalid_variant_raises():
    registry = load_registry(V2_REGISTRY)
    with pytest.raises(ValueError, match="Variant"):
        render_prompt_v2(registry, "Refuse-first", "vague", "v99")


def test_v2_prompt_hashes_are_unique():
    """All 36 prompts must have distinct hashes (no duplicates)."""
    registry = load_registry(V2_REGISTRY)
    hashes = set()
    for family in REQUIRED_FAMILIES:
        for cl in REQUIRED_CLARITY:
            for variant in registry[family]["clarity_levels"][cl]:
                text = render_prompt_v2(registry, family, cl, variant)
                h = get_prompt_hash(text)
                assert h not in hashes, f"Duplicate prompt hash for {family}/{cl}/{variant}"
                hashes.add(h)
    # 4 families × 3 clarity × 3 paraphrases = 36
    assert len(hashes) >= 36


def test_v2_total_prompt_count():
    """Exactly 4×3×3 = 36 prompts in registry."""
    registry = load_registry(V2_REGISTRY)
    count = 0
    for family in REQUIRED_FAMILIES:
        for cl in REQUIRED_CLARITY:
            count += len(registry[family]["clarity_levels"][cl])
    assert count == 36
