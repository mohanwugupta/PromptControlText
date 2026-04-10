import pytest
import os
from prompts.registry import load_registry, render_prompt, get_prompt_hash

def test_load_registry():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "registry_mock.yaml")
    registry = load_registry(fixture_path)
    
    assert "Refuse-first" in registry
    assert "Clarify-first" in registry
    assert "Hierarchy-first" in registry
    
    # Check that each family has at least 3 variants (PRD rule)
    for family, data in registry.items():
        assert len(data['variants']) >= 3

def test_render_prompt():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "registry_mock.yaml")
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
