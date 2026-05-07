import pytest
from models.client import LLMClient, ModelError

def test_llm_client_mock():
    client = LLMClient(mock_mode=True, mock_response="Mocked response.")
    
    prompt = "Hello"
    output, metadata = client.generate(
        system_prompt="You are a helper.",
        user_prompt=prompt,
        model="mock-model-v1",
        temperature=0.0
    )
    
    assert output == "Mocked response."
    assert metadata["model"] == "mock-model-v1"
    assert metadata["temperature"] == 0.0
    assert "timestamp" in metadata

def test_llm_client_caching():
    client = LLMClient(mock_mode=True, mock_response="Response 1", enable_cache=True)
    
    # First call
    out1, _ = client.generate("sys", "user", "mock", 0.0)
    assert out1 == "Response 1"
    
    # Change mock response to verify it's reading from cache, not mock
    client.mock_response = "Response 2"
    
    # Second call, should return cached Response 1
    out2, _ = client.generate("sys", "user", "mock", 0.0)
    assert out2 == "Response 1"
    
    # Different temp -> cache miss
    out3, _ = client.generate("sys", "user", "mock", 0.1)
    assert out3 == "Response 2"

def test_llm_client_error_handling():
    client = LLMClient(mock_mode=True, trigger_error=True)
    
    with pytest.raises(ModelError):
        client.generate("sys", "user", "mock", 0.0)

from models.vllm_client import VLLMClient
from unittest.mock import patch, MagicMock

def test_vllm_client_retries_and_fails():
    client = VLLMClient(max_retries=2, enable_cache=False, retry_delay=0.1)
    
    # Mock the internal property 'client' to raise an exception
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = Exception("Connection Timeout")
    
    # Inject it into local thread storage mimicking the property
    client._thread_local.openai_client = mock_openai_client
    
    with pytest.raises(ModelError, match="vLLM generation failed after 2 attempts"):
        client.generate("sys", "user", "mock", 0.0)
        
    assert mock_openai_client.chat.completions.create.call_count == 2

