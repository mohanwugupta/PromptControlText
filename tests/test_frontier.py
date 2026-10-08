import json
import io
import urllib.error
from collections import Counter

import pytest

from frontier.prepare import ROOT, digest, render_messages, sample, validate_manifest
from frontier.providers import generate, normalize, request_spec
from frontier.report import report
from frontier.run import Ledger, accounted_cost, cases, execute, reservation, validate_config


@pytest.fixture
def manifest():
    return json.loads((ROOT / "artifacts/frontier/manifest.json").read_text())


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/frontier-pilot.json").read_text())


def test_frozen_design_and_disjoint_pilot(manifest, config):
    validate_manifest(manifest)
    validate_config(config)
    main = list(cases(manifest, config, "main"))
    pilot = list(cases(manifest, config, "pilot"))
    assert len(main) == len({c["request_id"] for c in main}) == 10950
    assert len(pilot) == len({c["request_id"] for c in pilot}) == 150
    assert Counter(c["model"]["provider"] for c in main) == dict.fromkeys(["openai", "anthropic", "google"], 3650)
    assert {c["request_id"] for c in main}.isdisjoint(c["request_id"] for c in pilot)
    assert len({i["group_id"] for i in manifest["main"] + manifest["pilot"]}) == 60


def test_context_restored_and_control_retains_iheval_system(manifest):
    control = manifest["conditions"][0]
    for item in manifest["main"] + manifest["pilot"]:
        messages = render_messages(item, control)
        if item["benchmark"] == "iheval":
            assert messages[0] == {"role": "system", "content": item["reference"]["system_prompt"]}
            assert item["reference"]["access_code"] in messages[0]["content"]
        else:
            assert len(messages) == 1
        if item["benchmark"] == "harmbench" and item["reference"]["context"]:
            assert item["user_text"] == item["reference"]["context"] + "\n\n---\n\n" + item["reference"]["behavior"]


def test_only_intervention_changes_across_conditions(manifest):
    for item in manifest["main"]:
        for condition in manifest["conditions"]:
            messages = render_messages(item, condition)
            assert messages[-1] == {"role": "user", "content": item["user_text"]}
            if item["base_system"]:
                assert messages[0]["content"].startswith(item["base_system"])
            if condition["text"]:
                assert messages[0]["content"].endswith(condition["text"])


def test_manifest_tampering_and_rehashed_overlap_rejected(manifest):
    manifest["main"][0]["user_text"] += "changed"
    with pytest.raises(ValueError, match="hash"):
        validate_manifest(manifest)
    manifest["pilot"][0] = manifest["main"][0]
    manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(manifest)


def test_sampling_is_order_independent_and_excludes_parents_and_texts(manifest):
    population = manifest["main"] + manifest["pilot"]
    quotas = manifest["pilot_quotas"]
    first = sample(population, quotas, "seed")
    assert first == sample(list(reversed(population)), quotas, "seed")
    second = sample(population, quotas, "other", [i["group_id"] for i in first], [digest(i["user_text"]) for i in first])
    assert {i["group_id"] for i in first}.isdisjoint(i["group_id"] for i in second)


def test_all_adapters_preserve_system_and_user(config):
    messages = [{"role": "system", "content": "task\n\nintervention"}, {"role": "user", "content": "query"}]
    openai, anthropic, google = [request_spec(m, messages, 4096)[1] for m in config["models"]]
    assert openai["input"] == messages
    assert anthropic["system"] == google["systemInstruction"]["parts"][0]["text"] == messages[0]["content"]
    assert anthropic["messages"][0]["content"] == google["contents"][0]["parts"][0]["text"] == "query"
    for model in config["models"]:
        _, body = request_spec(model, messages[1:], 4096)
        assert "system" not in body and "systemInstruction" not in body
        assert "tools" not in body
    assert "temperature" not in openai and "temperature" not in anthropic


def test_request_identity_covers_configuration_and_messages(manifest, config):
    original = next(cases(manifest, config, "pilot"))["request_id"]
    config["max_output_tokens"] = 2048
    assert next(cases(manifest, config, "pilot"))["request_id"] != original
    config["max_output_tokens"] = 4096
    manifest["pilot"][0]["base_system"] += "alteration"
    changed = list(cases(manifest, config, "pilot"))
    original_manifest = json.loads((ROOT / "artifacts/frontier/manifest.json").read_text())
    original_cases = list(cases(original_manifest, config, "pilot"))
    assert sum(a["request_id"] != b["request_id"] for a, b in zip(changed, original_cases)) == 15


def test_openai_refusal_and_reasoning_usage():
    result = normalize("openai", {"status": "completed", "output": [
        {"type": "reasoning", "summary": []}, {"type": "message", "content": [{"type": "refusal", "refusal": "Declined"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 20}}})
    assert result["text"] == "Declined" and result["provider_refusal"]
    assert result["outcome"] == "text" and result["output_tokens"] == 30
    assert result["reasoning_tokens"] == 20


def test_anthropic_thinking_not_used_as_answer():
    result = normalize("anthropic", {"type": "message", "content": [
        {"type": "thinking", "thinking": "internal"}, {"type": "text", "text": "answer"}],
        "stop_reason": "max_tokens", "usage": {"input_tokens": 1, "output_tokens": 100, "cache_read_input_tokens": 9}})
    assert result["text"] == "answer" and result["truncated"]
    assert result["input_tokens"] == 10 and result["outcome"] == "truncated"


def test_google_thinking_counts_and_prompt_blocks():
    result = normalize("google", {"candidates": [{"content": {"parts": [
        {"thought": True, "text": "internal"}, {"text": "answer"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3, "thoughtsTokenCount": 20}})
    assert result["text"] == "answer" and result["output_tokens"] == 23
    block = normalize("google", {"promptFeedback": {"blockReason": "SAFETY"}})
    assert block["outcome"] == "blocked" and not block["provider_refusal"]


def test_unknown_usage_keeps_reservation(config):
    for model in config["models"]:
        cost, basis = accounted_cost({"outcome": "http_error"}, model, config)
        assert cost == reservation(model, config) and basis == "reserved_unknown"


def test_complete_pilot_and_resume_without_duplicate_calls(tmp_path, manifest, config, capsys):
    requests = list(cases(manifest, config, "pilot"))
    ledger = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    calls = []

    def fake(model, messages, config, environment):
        calls.append(model["provider"])
        return {"outcome": "text", "text": "mock output", "input_tokens": 10, "output_tokens": 10}

    assert execute(requests, ledger, config, {}, fake) == "complete"
    assert len(calls) == 150
    assert execute(requests, ledger, config, {}, fake) == "complete"
    assert len(calls) == 150
    exported = ledger.export(tmp_path / "results.jsonl")
    assert len(exported) == 150 and all(r["state"] == "done" for r in exported)
    assert ledger.total() < config["pilot_ceiling_usd"]


def test_budget_stops_before_network(tmp_path, manifest, config):
    config["pilot_ceiling_usd"] = 0.001
    ledger = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    requests = list(cases(manifest, config, "pilot"))

    def forbidden(*args):
        pytest.fail("Budget exhausted before any network request")

    assert execute(requests, ledger, config, {}, forbidden) == "budget_stopped"
    assert ledger.total() == 0 and not ledger.states()


def test_crash_keeps_reservation_and_prevents_blind_retry(tmp_path, manifest, config):
    requests = list(cases(manifest, config, "pilot"))
    path = tmp_path / "ledger.sqlite"
    ledger = Ledger(path, manifest, config)

    def crash(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(requests, ledger, config, {}, crash)
    reopened = Ledger(path, manifest, config)
    assert reopened.total() == reservation(requests[0]["model"], config)
    assert execute(requests, reopened, config, {}, crash).startswith("needs_review")


def test_provider_error_stops_and_is_exported(tmp_path, manifest, config, capsys):
    requests = list(cases(manifest, config, "pilot"))
    ledger = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    status = execute(requests, ledger, config, {}, lambda *args: {"outcome": "http_error", "http_status": 401})
    assert status.startswith("needs_review")
    records = ledger.export(tmp_path / "results.jsonl")
    assert len(records) == 1 and records[0]["result"]["http_status"] == 401
    assert records[0]["state"] == "needs_review" and records[0]["accounted_usd"] > 0


def test_ledger_refuses_different_configuration(tmp_path, manifest, config):
    path = tmp_path / "ledger.sqlite"
    Ledger(path, manifest, config)
    config["max_output_tokens"] = 2048
    with pytest.raises(ValueError, match="different"):
        Ledger(path, manifest, config)


@pytest.mark.parametrize("change", ["model", "cap", "price", "tokens"])
def test_configuration_guards(config, change):
    if change == "model":
        config["models"][0]["model"] = "unreviewed"
    elif change == "cap":
        config["pilot_ceiling_usd"] = 251
    elif change == "price":
        config["models"][0]["output_usd_per_million"] = float("nan")
    else:
        config["max_output_tokens"] = 128000
    with pytest.raises(ValueError):
        validate_config(config)


def test_report_reweights_strata_and_rejects_foreign_results(manifest, config):
    records = []
    for case in cases(manifest, config, "pilot"):
        # Different strata have different costs; pilot and main proportions differ.
        cost = 0.02 if case["item"]["stratum"] == "xstest_safe" else 0.01
        records.append({"request_id": case["request_id"], "state": "done", "accounted_usd": cost,
                        "result": {"outcome": "text"}})
    summary = report(records, manifest, config)
    assert summary["complete"]
    assert summary["projected_main_usd"] == pytest.approx(153.3)
    assert summary["projection_with_50_percent_headroom_usd"] == pytest.approx(229.95)
    records[0]["request_id"] = "foreign"
    with pytest.raises(ValueError, match="foreign"):
        report(records, manifest, config)


def test_partial_pilot_has_no_full_cost_projection(manifest, config):
    case = next(cases(manifest, config, "pilot"))
    summary = report([{"request_id": case["request_id"], "state": "reserved", "accounted_usd": 0.6,
                       "result": None}], manifest, config)
    assert not summary["complete"]
    assert summary["projected_main_usd"] is None
    assert summary["outcomes"] == {"unresolved": 1}


def test_infrastructure_is_inside_pilot_cap(config):
    config.update(pilot_ceiling_usd=10, runpod_reserve_usd=1, total_pilot_ceiling_usd=10)
    with pytest.raises(ValueError, match="infrastructure"):
        validate_config(config)


def test_blocked_provider_does_not_prevent_other_providers(tmp_path, manifest, config, capsys):
    requests = list(cases(manifest, config, "pilot"))
    ledger = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    anthropic = next(c for c in requests if c["model"]["provider"] == "anthropic")
    ledger.reserve(anthropic, reservation(anthropic["model"], config), config["pilot_ceiling_usd"])
    ledger.finish(anthropic, {"outcome": "http_error", "http_status": 400}, config)
    others = [c for c in requests if c["model"]["provider"] != "anthropic"]
    fake = lambda *args: {"outcome": "text", "text": "mock", "input_tokens": 10, "output_tokens": 10}
    assert execute(others, ledger, config, {}, fake) == "complete"
    assert list(ledger.states().values()).count("done") == 100
    assert ledger.states()[anthropic["request_id"]] == "needs_review"
    assert ledger.total() >= reservation(anthropic["model"], config)


def test_http_diagnostic_redacts_key(monkeypatch, config):
    secret = "test-secret-not-a-real-key"

    class FailingOpener:
        def open(self, request, timeout):
            body = json.dumps({"error": {"type": "invalid_request_error", "message": "Rejected " + secret}}).encode()
            raise urllib.error.HTTPError(request.full_url, 400, "error", {}, io.BytesIO(body))

    monkeypatch.setattr("frontier.providers.urllib.request.build_opener", lambda *args: FailingOpener())
    result = generate(config["models"][0], [{"role": "user", "content": "test"}], config,
                      {"OPENAI_API_KEY": secret})
    assert result["http_status"] == 400
    assert secret not in json.dumps(result)
    assert result["error_diagnostic"]["message"] == "Rejected [REDACTED]"


def test_partial_provider_report_does_not_invent_other_costs(manifest, config):
    records = [{"request_id": case["request_id"], "state": "done", "accounted_usd": 0.01,
                "result": {"outcome": "text"}}
               for case in cases(manifest, config, "pilot") if case["model"]["provider"] == "openai"]
    result = report(records, manifest, config)
    assert not result["complete"] and result["projected_main_usd"] is None
    assert result["completed_by_provider"] == {"openai": 50, "anthropic": 0, "google": 0}
    assert result["projected_main_usd_by_provider"] == {"openai": 36.5}
    records[0]["result"]["cost_basis"] = "reserved_unknown"
    result = report(records, manifest, config)
    assert result["projected_main_usd_by_provider"] == {}
    assert result["projected_main_accounting_usd_by_provider"] == {"openai": 36.5}


def test_known_http_policy_block_keeps_cost_and_continues(tmp_path, manifest, config, capsys):
    requests = [c for c in cases(manifest, config, "pilot") if c["model"]["provider"] == "openai"][:2]
    ledger = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    calls = []

    def fake(*args):
        calls.append(1)
        if len(calls) == 1:
            return {"outcome": "blocked", "block_origin": "provider_http", "error_diagnostic": {"code": "cyber_policy"}}
        return {"outcome": "text", "text": "mock", "input_tokens": 10, "output_tokens": 10}

    assert execute(requests, ledger, config, {}, fake) == "complete"
    assert len(calls) == 2 and all(s == "done" for s in ledger.states().values())
    assert ledger.total() >= reservation(requests[0]["model"], config)


def test_funded_retry_preserves_history_cost_and_completed_cases(tmp_path, manifest, config):
    selected = [c for c in cases(manifest, config, "pilot") if c["model"]["provider"] in ("openai", "anthropic")][:2]
    openai, anthropic = selected
    ledger = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    success = {"outcome": "text", "text": "mock", "input_tokens": 10, "output_tokens": 10}
    execute([openai], ledger, config, {}, lambda *args: dict(success))
    execute([anthropic], ledger, config, {}, lambda *args: {
        "outcome": "http_error", "http_status": 400,
        "error_diagnostic": {"message": "Your credit balance is too low"}})
    before = ledger.total()
    archived = ledger.retry_funded_anthropic(anthropic["request_id"], "User confirms credits added")
    assert ledger.total() == pytest.approx(before)
    interim = ledger.export(tmp_path / "interim.jsonl")
    assert report(interim, manifest, config)["accounted_pilot_usd"] == pytest.approx(before)
    assert any(r["state"] == "retry_ready" for r in interim)
    with pytest.raises(ValueError, match="No current failed"):
        ledger.retry_funded_anthropic(anthropic["request_id"], "Same command repeated")
    called = []
    def succeed(model, *args):
        called.append(model["provider"])
        return dict(success)
    reopened = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    assert execute(selected, reopened, config, {}, succeed) == "complete"
    assert called == ["anthropic"]
    records = reopened.export(tmp_path / "results.jsonl")
    summary = report(records, manifest, config)
    assert summary["prior_attempts"] == 1
    assert summary["prior_unknown_cost_reservations_usd"] == archived["accounted_usd"]
    assert summary["accounted_pilot_usd"] == pytest.approx(reopened.total())
    assert reopened.total() > before


def test_retry_reservation_cannot_evade_budget(tmp_path, manifest, config):
    case = next(c for c in cases(manifest, config, "pilot") if c["model"]["provider"] == "anthropic")
    ledger = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    execute([case], ledger, config, {}, lambda *args: {
        "outcome": "http_error", "http_status": 400,
        "error_diagnostic": {"message": "Your credit balance is too low"}})
    ledger.retry_funded_anthropic(case["request_id"], "User confirms funding")
    config["pilot_ceiling_usd"] = reservation(case["model"], config) * 1.5
    assert execute([case], ledger, config, {}, lambda *args: pytest.fail("Over-budget retry")) == "budget_stopped"


@pytest.mark.parametrize("outcome", [
    {"outcome": "http_error", "http_status": 400, "error_diagnostic": {"message": "Different failure"}},
    {"outcome": "transport_or_parse_error"},
    {"outcome": "text", "input_tokens": 10, "output_tokens": 10},
])
def test_funding_retry_rejects_other_outcomes(tmp_path, manifest, config, outcome):
    case = next(c for c in cases(manifest, config, "pilot") if c["model"]["provider"] == "anthropic")
    ledger = Ledger(tmp_path / "ledger.sqlite", manifest, config)
    execute([case], ledger, config, {}, lambda *args: outcome)
    before = ledger.total()
    with pytest.raises(ValueError, match="Only a known Anthropic credit rejection"):
        ledger.retry_funded_anthropic(case["request_id"], "Credits added")
    assert ledger.total() == before
