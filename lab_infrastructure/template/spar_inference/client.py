from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    seed: int = 0
    repetition_penalty: float = 1.05
    response_format: dict[str, Any] | None = None
    disable_thinking: bool = True


class VLLMClient:
    """Thread-safe OpenAI-compatible client adapted from PromptControlText."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 300.0,
        enable_cache: bool = False,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache: dict[str, str] = {}
        self._thread_local = threading.local()

    @property
    def client(self):
        if not hasattr(self._thread_local, "openai_client"):
            from openai import OpenAI
            self._thread_local.openai_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._thread_local.openai_client

    @staticmethod
    def _is_qwen_thinking_model(model: str) -> bool:
        name = model.lower()
        return "qwen3" in name or "qwq" in name

    def _cache_key(self, messages, model, config: GenerationConfig) -> str:
        payload = {"messages": messages, "model": model, "generation": asdict(config)}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        *,
        model: str | None = None,
        config: GenerationConfig | None = None,
    ) -> tuple[str, dict[str, Any]]:
        target_model = model or self.model_name
        config = config or GenerationConfig()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        cache_key = self._cache_key(messages, target_model, config)
        metadata = {
            "model": target_model,
            "base_url": self.base_url,
            "generation": asdict(config),
            "timestamp_unix": time.time(),
            "cached": False,
        }

        if self.enable_cache and cache_key in self.cache:
            metadata["cached"] = True
            return self.cache[cache_key], metadata

        request = {
            "model": target_model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "seed": config.seed,
        }

        if config.response_format is not None:
            request["response_format"] = config.response_format

        extra_body = {}
        if config.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = config.repetition_penalty
        if config.disable_thinking and self._is_qwen_thinking_model(target_model):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if extra_body:
            request["extra_body"] = extra_body

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                choice = response.choices[0]
                text = choice.message.content or ""
                metadata["finish_reason"] = choice.finish_reason
                metadata["attempt"] = attempt
                if response.usage is not None:
                    metadata["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                if self.enable_cache:
                    self.cache[cache_key] = text
                return text, metadata
            except Exception as exc:
                last_error = exc
                logger.warning("vLLM request failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise ModelError(
            f"vLLM generation failed after {self.max_retries} attempts: {last_error}"
        ) from last_error
