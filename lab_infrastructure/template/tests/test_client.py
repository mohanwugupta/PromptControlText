from spar_inference import GenerationConfig, VLLMClient


def test_cache_key_changes_with_generation_settings():
    client = VLLMClient("test-model")
    messages = [{"role": "user", "content": "hello"}]
    key_a = client._cache_key(messages, "test-model", GenerationConfig(max_tokens=32, seed=0))
    key_b = client._cache_key(messages, "test-model", GenerationConfig(max_tokens=64, seed=0))
    key_c = client._cache_key(messages, "test-model", GenerationConfig(max_tokens=32, seed=1))
    assert key_a != key_b
    assert key_a != key_c
