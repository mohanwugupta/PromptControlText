from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.schema import EvalItem
from experiments import run_phase1 as runner


@pytest.fixture
def generation_run(monkeypatch, tmp_path):
    items = [
        EvalItem(item_id=f"x_{i}", benchmark="XSTest", domain="safe",
                 input_text=f"Request {i}", gold_label="safe")
        for i in range(2)
    ]
    registry = {"Answer-first": {"clarity_levels": {"explicit": {"v1": "Answer."}}}}
    client = MagicMock()
    client.generate.return_value = ("New answer", {})
    monkeypatch.setattr(runner, "load_xstest", lambda _: items)
    monkeypatch.setattr(runner, "load_harmbench", lambda _: [])
    monkeypatch.setattr(runner, "load_iheval", lambda _: [])
    monkeypatch.setattr(runner, "load_registry", lambda _: registry)
    monkeypatch.setattr(runner, "LLMClient", lambda **_: client)
    path = tmp_path / "responses.csv"

    saved_client = MagicMock()
    saved_client.generate.return_value = ("Saved answer\nsecond line", {"old": True})
    saved = runner._generate_one(
        saved_client, items[0], "Answer-first", "v1", "Answer.", "test-model", "explicit"
    )
    pd.DataFrame([saved]).to_csv(path, index=False)

    def run(max_workers=1):
        runner.run_experiment(path, generator_model="test-model", mock_mode=True,
                              max_workers=max_workers, resume=True)

    return path, saved, client, run


def test_resume_preserves_saved_responses_and_skips_complete_run(generation_run):
    path, saved, client, run = generation_run
    run()
    rows = pd.read_csv(path, keep_default_na=False).to_dict(orient="records")
    assert len(rows) == 2
    assert rows[0]["model_output"] == saved["model_output"]
    assert rows[1]["model_output"] == "New answer"
    assert client.generate.call_count == 1
    assert client.generate.call_args.kwargs["user_prompt"] == "Request 1"
    assert not path.with_suffix(".csv.tmp").exists()

    complete_checkpoint = path.read_bytes()
    run()
    assert client.generate.call_count == 1
    assert path.read_bytes() == complete_checkpoint


def test_resume_retries_empty_saved_outputs(generation_run):
    path, saved, client, run = generation_run
    saved["model_output"] = "  "
    pd.DataFrame([saved]).to_csv(path, index=False)
    run()
    assert client.generate.call_count == 2
    rows = pd.read_csv(path)
    assert len(rows) == 2
    assert set(rows["model_output"]) == {"New answer"}


@pytest.mark.parametrize("field,value", [
    ("model_name", "different-model"),
    ("input_text", "Changed benchmark input"),
    ("prompt_variant", "unknown-variant"),
])
def test_resume_rejects_mismatched_checkpoint_without_overwriting(generation_run, field, value):
    path, saved, client, run = generation_run
    saved[field] = value
    pd.DataFrame([saved]).to_csv(path, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Cannot resume"):
        run()
    client.generate.assert_not_called()
    assert path.read_bytes() == original


def test_resume_rejects_duplicate_conditions(generation_run):
    path, saved, client, run = generation_run
    pd.DataFrame([saved, saved]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate checkpoint condition"):
        run()
    client.generate.assert_not_called()


def test_failed_generation_preserves_checkpoint_and_reports_incomplete(generation_run):
    path, saved, client, run = generation_run
    client.generate.side_effect = RuntimeError("Endpoint unavailable")
    with pytest.raises(RuntimeError, match="Generation incomplete: 1/2"):
        run()
    rows = pd.read_csv(path)
    assert len(rows) == 1
    assert rows.iloc[0]["model_output"] == saved["model_output"]


def test_interrupted_checkpoint_write_preserves_previous_file(generation_run, monkeypatch):
    path, _, _, run = generation_run
    original = path.read_bytes()

    def interrupted_replace(*_):
        raise OSError("Interrupted before atomic replacement")

    monkeypatch.setattr(runner.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="Interrupted"):
        run()
    assert path.read_bytes() == original


def test_dead_server_stops_remaining_requests_and_saves_new_progress(generation_run, monkeypatch):
    from models.vllm_client import ServerUnavailableError

    path, saved, client, run = generation_run
    items = [EvalItem(item_id=f"x_{i}", benchmark="XSTest", domain="safe",
                      input_text=f"Request {i}", gold_label="safe") for i in range(30)]
    monkeypatch.setattr(runner, "load_xstest", lambda _: items)
    client.generate.side_effect = [
        ("New answer", {}),
        ServerUnavailableError("Endpoint disconnected after retries"),
        AssertionError("Remaining conditions should not be submitted"),
    ]
    with pytest.raises(RuntimeError, match="server unreachable: saved 2/30"):
        run()
    assert client.generate.call_count == 2
    rows = pd.read_csv(path, keep_default_na=False)
    assert len(rows) == 2
    assert rows.iloc[0]["model_output"] == saved["model_output"]
    assert rows.iloc[1]["model_output"] == "New answer"

    client.reset_mock()
    client.generate.side_effect = None
    run()
    assert client.generate.call_count == 28
    rows = pd.read_csv(path)
    assert len(rows) == 30
    assert rows["item_id"].nunique() == 30


def test_dead_server_retains_successful_inflight_response(generation_run, monkeypatch):
    from threading import Event
    from models.vllm_client import ServerUnavailableError

    path, _, client, run = generation_run
    items = [EvalItem(item_id=f"x_{i}", benchmark="XSTest", domain="safe",
                      input_text=f"Request {i}", gold_label="safe") for i in range(30)]
    monkeypatch.setattr(runner, "load_xstest", lambda _: items)
    failure_observed = Event()
    original_wait = runner.wait

    def wait_and_release_inflight(*args, **kwargs):
        result = original_wait(*args, **kwargs)
        failure_observed.set()
        return result

    def respond(**kwargs):
        if kwargs["user_prompt"] == "Request 1":
            assert failure_observed.wait(timeout=5)
            return "Successful in-flight answer", {}
        raise ServerUnavailableError("Endpoint unavailable")

    monkeypatch.setattr(runner, "wait", wait_and_release_inflight)
    client.generate.side_effect = respond
    with pytest.raises(RuntimeError, match="server unreachable: saved 2/30"):
        run(max_workers=2)
    assert client.generate.call_count == 2
    rows = pd.read_csv(path)
    assert len(rows) == 2
    assert rows.iloc[1]["model_output"] == "Successful in-flight answer"
