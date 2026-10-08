import copy
import json

import pytest

from frontier.batch_api import ApiError, Client, result_row, terminal
from frontier.main_run import (MainLedger, batch_rows, cost_result, main_cases,
                               reserve_cost, step, validate_main)
from frontier.prepare import ROOT
from frontier.providers import request_spec


@pytest.fixture
def setup(tmp_path):
    manifest = json.loads((ROOT / "artifacts/frontier/manifest.json").read_text())
    config = json.loads((ROOT / "configs/frontier-main.json").read_text())
    pilot = json.loads((ROOT / "configs/frontier-pilot.json").read_text())
    validate_main(config, manifest, pilot)
    requests = list(main_cases(manifest, config))
    ledger = MainLedger(tmp_path / "main.sqlite", manifest, config)
    return manifest, config, pilot, requests, ledger, tmp_path


def test_main_design_and_exact_generation_settings(setup):
    _, config, _, requests, _, _ = setup
    assert len(requests) == len({c["request_id"] for c in requests}) == 10950
    for provider in ("openai", "anthropic", "google"):
        selected = [c for c in requests if c["model"]["provider"] == provider]
        assert len(selected) == 3650
        assert len({c["item"]["item_id"] for c in selected}) == 50
        assert len({c["condition"]["condition_id"] for c in selected}) == 73
    for case in requests:
        original = request_spec(case["model"], case["messages"], config["max_output_tokens"])[1]
        original.pop("service_tier", None)
        assert original == case["body"]


@pytest.mark.parametrize("field,value", [("main_generation_ceiling_usd", 161), ("study_ceiling_usd", 251),
    ("prior_pilot_accounted_usd", 0), ("max_output_tokens", 2048), ("batch_discount", .25),
    ("judge_allocation_usd", 0), ("main_runpod_reserve_usd", 0), ("initial_batch_size", 200)])
def test_main_cannot_silently_change_budget_or_design(setup, field, value):
    manifest, config, pilot, *_ = setup
    config[field] = value
    with pytest.raises(ValueError):
        validate_main(config, manifest, pilot)


def test_full_batch_reserved_before_submission_and_duplicates_atomic(setup):
    _, config, _, requests, ledger, _ = setup
    selected = [c for c in requests if c["model"]["provider"] == "openai"][:200]
    key = ledger.prepare("openai", selected)
    assert ledger.total() == pytest.approx(200 * .3072)
    assert ledger.db.execute("SELECT state FROM batches").fetchone()[0] == "submitting"
    with pytest.raises(Exception):
        ledger.prepare("openai", selected)
    assert ledger.db.execute("SELECT count(*) FROM requests").fetchone()[0] == 200
    assert ledger.total() == pytest.approx(61.44)
    ledger.prepare("openai", [c for c in requests if c["model"]["provider"] == "openai"][200:400])
    before = ledger.total()
    assert ledger.prepare("openai", [c for c in requests if c["model"]["provider"] == "openai"][400:600]) is None
    assert ledger.total() == before


class FakeClient:
    def __init__(self, ledger, fail=False):
        self.ledger, self.fail, self.calls = ledger, fail, []

    def upload_openai(self, rows, key):
        assert self.ledger.total() > 0
        return "file-test"

    def submit(self, provider, rows, key, model, file_id):
        assert self.ledger.total() > 0
        self.calls.append((provider, rows))
        if self.fail:
            raise ApiError("ambiguous timeout")
        return {"id": key, "name": "batches/" + key}

    def poll(self, provider, remote_id):
        return {"status": "in_progress", "processing_status": "in_progress", "done": False}


def test_initial_batches_and_crash_resume_no_duplicate(setup):
    manifest, config, _, requests, ledger, path = setup
    client = FakeClient(ledger)
    assert step(ledger, requests, client, path) == "running"
    assert len(client.calls) == 3
    assert all(len(rows) == 5 for _, rows in client.calls)
    assert ledger.total() == pytest.approx(2.43712)
    resumed = MainLedger(path / "main.sqlite", manifest, config)
    assert step(resumed, requests, client, path) == "running"
    assert len(client.calls) == 3


def test_ambiguous_post_stops_all_new_submissions(setup):
    *_, requests, ledger, path = setup
    client = FakeClient(ledger, fail=True)
    assert step(ledger, requests, client, path) == "needs_review"
    assert len(client.calls) == 1
    before = ledger.total()
    assert step(ledger, requests, client, path) == "needs_review"
    assert len(client.calls) == 1 and ledger.total() == before


def openai_success(key):
    return {"custom_id": key, "response": {"status_code": 200, "body": {
        "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
        "usage": {"input_tokens": 50, "output_tokens": 25}}}}


def test_ingest_matches_ids_out_of_order_and_releases_known_usage(setup):
    _, config, _, requests, ledger, path = setup
    selected = [c for c in requests if c["model"]["provider"] == "openai"][:5]
    key = ledger.prepare("openai", selected)
    batch = dict(ledger.db.execute("SELECT * FROM batches").fetchone())
    rows = [openai_success(c["request_id"]) for c in reversed(selected)]
    ledger.ingest(batch, b"\n".join(json.dumps(row).encode() for row in rows))
    assert ledger.summary()["done_by_provider"] == {"openai": 5}
    assert ledger.total() == pytest.approx(5 * (50 * 12.5 + 25 * 50) / 2e6)
    ledger.checkpoint(path)
    assert len((path / "results.jsonl").read_text().splitlines()) == 5
    assert (path / "checkpoint.sqlite").exists()
    # Re-reading the same immutable final results does not duplicate charges.
    before = ledger.total()
    ledger.ingest(batch, b"\n".join(json.dumps(row).encode() for row in rows))
    assert ledger.total() == before


@pytest.mark.parametrize("defect", ["duplicate", "foreign", "missing"])
def test_bad_result_identity_retains_every_reservation(setup, defect):
    *_, requests, ledger, _ = setup
    selected = [c for c in requests if c["model"]["provider"] == "openai"][:2]
    ledger.prepare("openai", selected)
    batch = dict(ledger.db.execute("SELECT * FROM batches").fetchone())
    rows = [openai_success(c["request_id"]) for c in selected]
    if defect == "duplicate": rows.append(rows[0])
    if defect == "foreign": rows[0]["custom_id"] = "foreign"
    if defect == "missing": rows.pop()
    before = ledger.total()
    with pytest.raises(ValueError):
        ledger.ingest(batch, b"\n".join(json.dumps(row).encode() for row in rows))
    assert ledger.total() == before


def test_blocks_separate_from_refusal_and_unknown_errors_retained(setup):
    _, config, _, requests, _, _ = setup
    case = requests[0]
    _, blocked = result_row("openai", {"custom_id": "x", "response": {"status_code": 400,
        "body": {"error": {"code": "cyber_policy"}}}})
    assert blocked["outcome"] == "blocked" and not blocked["provider_refusal"]
    assert cost_result(case, blocked, config)[0] == 0
    _, unknown = result_row("openai", {"custom_id": "x", "error": {"code": "server_error"}})
    assert cost_result(case, unknown, config) == (reserve_cost(case, config), "reserved_unknown")
    _, anthropic = result_row("anthropic", {"custom_id": "x", "result": {"type": "succeeded", "message": {
        "content": [], "stop_reason": "refusal", "usage": {"input_tokens": 50, "output_tokens": 0}}}})
    assert anthropic["outcome"] == "blocked"
    assert cost_result(case, anthropic, config)[0] > 0


def test_google_rest_response_identity_and_thinking_accounting():
    row = {"metadata": {"key": "x"}, "response": {"candidates": [{"content": {"parts": [{"text": "answer"}]},
        "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "thoughtsTokenCount": 30}}}
    key, result = result_row("google", row)
    assert key == "x" and result["output_tokens"] == 50
    client = Client({"GEMINI_API_KEY": "test"})
    status = {"done": True, "response": {"inlinedResponses": {"inlinedResponses": [row]}}}
    assert terminal("google", status)
    assert json.loads(client.results("google", "batches/test", status)) == row
    with pytest.raises(ValueError):
        client.results("google", "batches/test", {"done": True, "error": {}})


def test_poll_only_never_submits_and_get_failure_does_not_resubmit(setup):
    *_, requests, ledger, path = setup
    client = FakeClient(ledger)
    assert step(ledger, requests, client, path, submit=False) == "polled"
    assert not client.calls
    step(ledger, requests, client, path)
    def fail(*args): raise ApiError("network")
    client.poll = fail
    before = ledger.total()
    assert step(ledger, requests, client, path) == "poll_error"
    assert len(client.calls) == 3 and ledger.total() == before


def test_changed_config_cannot_reopen_ledger(setup):
    manifest, config, _, _, _, path = setup
    config["batch_size"] = 201
    with pytest.raises(ValueError):
        MainLedger(path / "main.sqlite", manifest, config)
