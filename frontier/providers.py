"""Minimal, explicit HTTP adapters. No tools, prompt rewrites, or hidden retries."""

import json
import urllib.error
import urllib.request

KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}
MODEL_IDS = {"openai": "gpt-6-astra", "anthropic": "claude-opus-5-5",
             "google": "gemini-3.1-pro-preview"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_spec(model, messages, max_output_tokens):
    provider, name = model["provider"], model["model"]
    if MODEL_IDS.get(provider) != name:
        raise ValueError("Unreviewed model/adapter combination")
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user = messages[-1]["content"]
    if provider == "openai":
        return "https://api.openai.com/v1/responses", {
            "model": name, "input": messages, "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": model["effort"]}, "store": False, "service_tier": "default",
        }
    if provider == "anthropic":
        body = {"model": name, "messages": [{"role": "user", "content": user}],
                "max_tokens": max_output_tokens, "output_config": {"effort": model["effort"]}}
        if system:
            body["system"] = system
        return "https://api.anthropic.com/v1/messages", body
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_output_tokens, "candidateCount": 1,
                                 "temperature": 1.0, "thinkingConfig": {"thinkingLevel": model["effort"]}}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent", body


def normalize(provider, data):
    """Keep transport/filter outcomes separate from the later six-policy judge."""
    result = {"text": "", "outcome": "empty", "input_tokens": None, "output_tokens": None,
              "reasoning_tokens": None, "truncated": False, "provider_refusal": False}
    if provider == "openai":
        blocks = [b for o in data.get("output", []) if o.get("type") == "message"
                  for b in o.get("content", [])]
        result["text"] = "".join(b.get("text", b.get("refusal", "")) for b in blocks
                                   if b.get("type") in ("output_text", "refusal"))
        result["provider_refusal"] = any(b.get("type") == "refusal" for b in blocks)
        usage = data.get("usage") or {}
        result.update(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                      reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                      finish_reason=data.get("status"), incomplete_details=data.get("incomplete_details"),
                      response_id=data.get("id"), returned_model=data.get("model"),
                      usage=usage, service_tier=data.get("service_tier"))
        result["truncated"] = (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
        blocked = (data.get("incomplete_details") or {}).get("reason") == "content_filter"
        failed = data.get("status") not in ("completed", "incomplete")
    elif provider == "anthropic":
        result["text"] = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        if input_tokens is not None:
            input_tokens += usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
        result.update(input_tokens=input_tokens, output_tokens=usage.get("output_tokens"),
                      reasoning_tokens=(usage.get("output_tokens_details") or {}).get("thinking_tokens"),
                      finish_reason=data.get("stop_reason"), response_id=data.get("id"),
                      returned_model=data.get("model"), usage=usage, stop_details=data.get("stop_details"))
        result["truncated"] = data.get("stop_reason") in ("max_tokens", "model_context_window_exceeded")
        result["provider_refusal"] = data.get("stop_reason") == "refusal"
        blocked, failed = result["provider_refusal"], data.get("type") == "error"
        if blocked:
            result["block_origin"] = "provider_stop_reason"
    else:
        candidates = data.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        result["text"] = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", [])
                                   if not p.get("thought", False))
        usage = data.get("usageMetadata") or {}
        output = usage.get("candidatesTokenCount")
        # Gemini reports thinking separately; those tokens are also billed output.
        if output is not None:
            output += usage.get("thoughtsTokenCount", 0)
        result.update(input_tokens=usage.get("promptTokenCount"), output_tokens=output,
                      reasoning_tokens=usage.get("thoughtsTokenCount"), usage=usage,
                      finish_reason=candidate.get("finishReason"), response_id=data.get("responseId"),
                      returned_model=data.get("modelVersion"), prompt_feedback=data.get("promptFeedback"),
                      safety_ratings=candidate.get("safetyRatings"))
        result["truncated"] = candidate.get("finishReason") == "MAX_TOKENS"
        blocked = bool((data.get("promptFeedback") or {}).get("blockReason")) or candidate.get("finishReason") in (
            "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY")
        failed = "error" in data
    result["outcome"] = ("provider_error" if failed else "blocked" if blocked else "truncated"
                         if result["truncated"] else "text" if result["text"] else "empty")
    return result


def generate(model, messages, config, environment):
    url, body = request_spec(model, messages, config["max_output_tokens"])
    provider = model["provider"]
    key = environment[KEYS[provider]]
    headers = {"Content-Type": "application/json", "User-Agent": "PromptControlText-frontier/1"}
    if provider == "openai":
        headers["Authorization"] = "Bearer " + key
    elif provider == "anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        if environment.get("ANTHROPIC_WORKSPACE_ID"):
            headers["anthropic-workspace-id"] = environment["ANTHROPIC_WORKSPACE_ID"]
    else:
        headers["x-goog-api-key"] = key
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    opener = urllib.request.build_opener(NoRedirect)
    # Never return request headers, credentials, or arbitrary provider error bodies.
    try:
        with opener.open(request, timeout=config["timeout_seconds"]) as response:
            result = normalize(provider, json.load(response))
            result["http_request_id"] = response.headers.get("x-request-id") or response.headers.get("request-id")
            return result
    except urllib.error.HTTPError as error:
        diagnostic = {}
        try:
            payload = json.loads(error.read()).get("error", {})
            if isinstance(payload, dict):
                diagnostic = {name: str(payload[name]).replace(key, "[REDACTED]")[:1000]
                              for name in ("type", "code", "status", "message") if name in payload}
        except (ValueError, OSError, AttributeError):
            pass
        policy_block = provider == "openai" and error.code == 400 and diagnostic.get("code") == "cyber_policy"
        return {"outcome": "blocked" if policy_block else "http_error", "http_status": error.code,
                "http_request_id": error.headers.get("x-request-id") or error.headers.get("request-id"),
                "error_diagnostic": diagnostic,
                "block_origin": "provider_http" if policy_block else None,
                "text": "", "provider_refusal": False}
    except (OSError, ValueError, KeyError, TypeError):
        return {"outcome": "transport_or_parse_error"}
