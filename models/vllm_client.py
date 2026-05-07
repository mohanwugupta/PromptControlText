import time
import hashlib
import threading
import logging
from typing import Dict, Tuple, Any
from models.client import ModelError

logger = logging.getLogger(__name__)

class VLLMClient:
    """
    Client for talking to a vLLM server via the OpenAI API.
    Replaces the mock LLMClient with actual cluster generation capabilities.
    """
    def __init__(
        self, 
        model_name: str = "Qwen2.5-72B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 300.0,
        enable_cache: bool = True
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache: Dict[str, str] = {}
        self._thread_local = threading.local()

    def _get_cache_key(self, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
        key_content = f"{system_prompt}|{user_prompt}|{model}|{temperature}"
        return hashlib.sha256(key_content.encode('utf-8')).hexdigest()

    @property
    def client(self):
        # Each thread gets its own OpenAI client
        if not hasattr(self._thread_local, "openai_client"):
            from openai import OpenAI
            self._thread_local.openai_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._thread_local.openai_client

    def generate(self, system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.0, max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:
        target_model = model or self.model_name
        metadata = {
            "model": target_model,
            "temperature": temperature,
            "timestamp": time.time(),
        }

        cache_key = self._get_cache_key(system_prompt, user_prompt, target_model, temperature)
        if self.enable_cache and cache_key in self.cache:
            return self.cache[cache_key], metadata

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                text = choice.message.content or ""
                
                metadata["finish_reason"] = choice.finish_reason
                if response.usage:
                    metadata["prompt_tokens"] = response.usage.prompt_tokens
                    metadata["completion_tokens"] = response.usage.completion_tokens

                if self.enable_cache:
                    self.cache[cache_key] = text

                return text, metadata

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "vLLM request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise ModelError(
            f"vLLM generation failed after {self.max_retries} attempts: {last_error}"
        ) from last_error
