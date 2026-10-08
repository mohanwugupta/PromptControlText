"""Provider Batch APIs; explicit requests, no redirects or automatic retries."""
import json
import re
import urllib.error
import urllib.request
import uuid

from frontier.providers import KEYS, NoRedirect, normalize

BASE = {"openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta"}


class ApiError(RuntimeError):
    pass


class Client:
    def __init__(self, environment, timeout=180):
        self.environment, self.timeout = environment, timeout
        self.opener = urllib.request.build_opener(NoRedirect)

    def request(self, provider, path, method="GET", data=None, content_type="application/json", raw=False):
        if not path.startswith("/") or ".." in path or "://" in path:
            raise ValueError("Invalid provider path")
        key = self.environment[KEYS[provider]]
        headers = {"Content-Type": content_type, "User-Agent": "PromptControlText-frontier/2"}
        if provider == "openai":
            headers["Authorization"] = "Bearer " + key
        elif provider == "anthropic":
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
            if self.environment.get("ANTHROPIC_WORKSPACE_ID"):
                headers["anthropic-workspace-id"] = self.environment["ANTHROPIC_WORKSPACE_ID"]
        else:
            headers["x-goog-api-key"] = key
        if data is not None and not isinstance(data, bytes):
            data = json.dumps(data).encode()
        req = urllib.request.Request(BASE[provider] + path, data=data, headers=headers, method=method)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                body = response.read()
                return body if raw else json.loads(body)
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="replace")[:4000]
            for secret in self.environment.values():
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            raise ApiError(f"{provider} HTTP {error.code}: {message}") from None
        except (OSError, ValueError) as error:
            raise ApiError(f"{provider} {type(error).__name__}; outcome may be ambiguous") from None

    def upload_openai(self, rows, batch_key):
        boundary = "pct" + uuid.uuid4().hex
        content = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
        data = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{batch_key}.jsonl"\r\n'
                'Content-Type: application/jsonl\r\n\r\n').encode() + content
        data += f"\r\n--{boundary}--\r\n".encode()
        return self.request("openai", "/files", "POST", data,
                            "multipart/form-data; boundary=" + boundary)["id"]

    def submit(self, provider, rows, batch_key, model, file_id=None):
        if provider == "openai":
            return self.request(provider, "/batches", "POST", {
                "input_file_id": file_id, "endpoint": "/v1/responses", "completion_window": "24h",
                "metadata": {"experiment": "PromptControlText-main", "batch_key": batch_key}})
        if provider == "anthropic":
            return self.request(provider, "/messages/batches", "POST", {"requests": rows})
        return self.request(provider, f"/models/{model}:batchGenerateContent", "POST", {
            "batch": {"display_name": "PromptControlText-" + batch_key,
                      "input_config": {"requests": {"requests": rows}}}})

    def poll(self, provider, remote_id):
        if not re.fullmatch(r"[A-Za-z0-9_/-]+", remote_id):
            raise ValueError("Invalid remote batch ID")
        prefix = "/messages/batches/" if provider == "anthropic" else "/batches/"
        path = "/" + remote_id if provider == "google" else prefix + remote_id
        return self.request(provider, path)

    def results(self, provider, remote_id, status):
        if provider == "openai":
            chunks = []
            for field in ("output_file_id", "error_file_id"):
                file_id = status.get(field)
                if file_id:
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
                        raise ValueError("Invalid results file ID")
                    chunks.append(self.request(provider, f"/files/{file_id}/content", raw=True))
            return b"\n".join(chunks)
        if provider == "anthropic":
            return self.request(provider, f"/messages/batches/{remote_id}/results", raw=True)
        # REST returns Operation.response, unlike the SDK's dest.inlined_responses.
        response = status.get("response") or {}
        rows = (response.get("inlinedResponses") or {}).get("inlinedResponses")
        if rows is None:
            raise ValueError("Missing Gemini inline results; retain reservations")
        return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def terminal(provider, status):
    if provider == "openai":
        return status.get("status") in ("completed", "failed", "expired", "cancelled")
    if provider == "anthropic":
        return status.get("processing_status") == "ended"
    return status.get("done") is True


def result_row(provider, row):
    """Return identity and normalized result; never infer identity from row order."""
    if provider == "google":
        key = (row.get("metadata") or {}).get("key")
        if "response" in row:
            return key, normalize(provider, row["response"])
        return key, {"outcome": "batch_error", "text": "", "batch_error": row.get("error")}
    key = row.get("custom_id")
    if provider == "anthropic":
        result = row["result"]
        if result["type"] == "succeeded":
            return key, normalize(provider, result["message"])
        return key, {"outcome": "batch_error", "text": "", "batch_error": result,
                     "documented_unbilled": result["type"] in ("errored", "canceled", "expired")}
    response = row.get("response") or {}
    body = response.get("body") or {}
    if response.get("status_code") == 200:
        result = normalize(provider, body)
        result["http_request_id"] = response.get("request_id")
        return key, result
    error = body.get("error") or row.get("error") or {}
    policy_block = response.get("status_code") == 400 and error.get("code") == "cyber_policy"
    # Batch bills completed work. This explicit pre-inference validation rejection
    # is recorded separately from transport/server errors (which retain reserves).
    return key, {"outcome": "blocked" if policy_block else "batch_error", "text": "",
                 "block_origin": "provider_http" if policy_block else None,
                 "http_status": response.get("status_code"), "error_diagnostic": error,
                 "http_request_id": response.get("request_id"), "provider_refusal": False,
                 "documented_unbilled": policy_block or error.get("code") == "batch_expired"}
