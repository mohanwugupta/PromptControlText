import os

import pytest

from prompts.registry import (
    detect_schema_version,
    get_prompt_hash,
    iter_prompt_triples,
    list_prompt_configs,
    load_registry,
    render_prompt_v2,
    render_prompt_v3,
    validate_registry_schema,
)


FIXTURE_V3 = os.path.join(os.path.dirname(__file__), "fixtures", "registry_v3_mock.yaml")
REGISTRY_V3 = os.path.join(os.path.dirname(__file__), "..", "prompts", "registry_v3.yaml")
REGISTRY_CONTROL = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "registry_control.yaml"
)


def test_prompt_hash_stability():
    assert get_prompt_hash("A prompt") == get_prompt_hash("A prompt")
    assert get_prompt_hash("A prompt") != get_prompt_hash("A prompt ")


def test_detect_v3_schema():
    assert detect_schema_version(load_registry(FIXTURE_V3)) == "v2"


def test_detect_empty_schema_raises():
    with pytest.raises(ValueError, match="empty"):
        detect_schema_version({})


def test_validate_v3_registries():
    validate_registry_schema(load_registry(FIXTURE_V3))
    validate_registry_schema(load_registry(REGISTRY_V3))


def test_validate_bad_clarity_level_raises():
    registry = {
        "TestFamily": {
            "clarity_levels": {
                "invalid_level": {"v1": "x", "v2": "y", "v3": "z"}
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


def test_v3_registry_matches_paper_design():
    registry = load_registry(REGISTRY_V3)
    assert set(registry) == {
        "Answer-first",
        "Refuse-first",
        "Clarify-first",
        "Minimal-safe-help",
        "Transform-classify-first",
        "Source-isolation-first",
        "Hierarchy-first",
        "Evidence-first",
    }
    assert len(list_prompt_configs(registry)) == 72


def test_v3_registry_cells_are_complete_and_unique():
    registry = load_registry(REGISTRY_V3)
    prompt_hashes = []
    for family, family_data in registry.items():
        assert set(family_data["clarity_levels"]) == {
            "vague",
            "explicit",
            "explicit_fallback",
        }
        for variants in family_data["clarity_levels"].values():
            assert set(variants) == {"v1", "v2", "v3"}
        for _, _, _, text in iter_prompt_triples({family: family_data}):
            assert text.strip()
            prompt_hashes.append(get_prompt_hash(text))
    assert len(prompt_hashes) == len(set(prompt_hashes)) == 72


def test_render_aliases_match():
    registry = load_registry(REGISTRY_V3)
    assert render_prompt_v2(registry, "Hierarchy-first", "vague", "v2") == (
        render_prompt_v3(registry, "Hierarchy-first", "vague", "v2")
    )


@pytest.mark.parametrize(
    "family,clarity,variant",
    [
        ("Missing-family", "vague", "v1"),
        ("Answer-first", "missing", "v1"),
        ("Answer-first", "vague", "v99"),
    ],
)
def test_render_unknown_config_raises(family, clarity, variant):
    registry = load_registry(REGISTRY_V3)
    with pytest.raises(ValueError, match="not found"):
        render_prompt_v3(registry, family, clarity, variant)


def test_iter_prompt_triples_matches_rendering():
    registry = load_registry(REGISTRY_V3)
    triples = list(iter_prompt_triples(registry))
    assert len(triples) == 72
    assert {clarity for _, clarity, _, _ in triples} == {
        "vague",
        "explicit",
        "explicit_fallback",
    }
    for family, clarity, variant, text in triples:
        assert render_prompt_v3(registry, family, clarity, variant) == text


def test_control_registry_is_one_empty_system_prompt():
    registry = load_registry(REGISTRY_CONTROL)
    triples = list(iter_prompt_triples(registry))
    assert triples == [("No-system-prompt", None, "v1", None)]


def test_prompt_configs_are_sorted():
    configs = list_prompt_configs(load_registry(REGISTRY_V3))
    assert configs == sorted(configs)
