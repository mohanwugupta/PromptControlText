"""Offline conversion, request-safe resumes, and the cluster judge join."""
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from frontier import prepare_judge as prep
from frontier.finish_judge import finish
from models.vllm_client import ServerUnavailableError
from scoring import llm_policy_runner as runner
from scoring.llm_policy_adjudicate import adjudicate_row
from scoring.llm_policy_judge import judge_row
from scoring.llm_policy_run_jobs import validate_job_input


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def upload(tmp_path, monkeypatch):
    """Small frozen main design with duplicate text and provider blocks."""
    models = [{"provider": p, "model": p + "-model"} for p in ("openai", "anthropic", "google")]
    config = {"models": models, "canonical_judge": "meta-llama/Llama-3.1-8B-Instruct"}
    cases, records, metadata = [], [], []
    for model in models:
        for index in range(3):
            text = "  Repeated answer,\nquoted \"text\".  " if index == 0 else ""
            if index == 2:
                text = "I cannot answer."  # Provider flagged a block but supplied judgeable text.
            key = model["provider"] + str(index)
            case = {"request_id": key, "model": model,
                    "request_body_sha256": hashlib.sha256(key.encode()).hexdigest(),
                    "item": {"item_id": "item", "group_id": "group", "benchmark": "iheval",
                             "stratum": "safety", "domain": "qa", "gold_label": "reference",
                             "user_text": "TASK METADATA MUST NOT REACH JUDGE", "base_system": "base",
                             "reference": {"answers": ["correct"]}},
                    "condition": {"condition_id": str(index), "family": "family" + str(index),
                                  "clarity": None if index == 0 else "explicit", "variant": "v1",
                                  "text": None if index == 0 else "system"}}
            result = {"text": text, "outcome": "text" if index == 0 else "blocked",
                      "provider_refusal": index != 0, "truncated": False}
            cases.append(case)
            records.append({"request_id": key, "provider": model["provider"], "state": "done",
                            "case": case, "result": result})
            metadata.append({"request_id": key, "state": "done", "provider": model["provider"],
                             "item_id": "item", "condition_id": str(index), "outcome": result["outcome"],
                             "request_body_sha256": case["request_body_sha256"],
                             "response_text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    run = tmp_path / "run"
    (run / "responses").mkdir(parents=True)
    write_jsonl(run / "responses/batch.jsonl", records)
    write_jsonl(run / "requests.jsonl", metadata)
    (run / "progress.json").write_text("{}")
    paths = []
    for name, data in [("manifest", {"manifest_sha256": "frozen"}), ("config", config), ("pilot", {})]:
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(data))
        paths.append(path)
    monkeypatch.setattr(prep, "validate_main", lambda config, *_: config)
    monkeypatch.setattr(prep, "main_cases", lambda *_: iter(cases))
    out = tmp_path / "prepared"
    return {"run": run, "out": out, "paths": paths, "records": records,
            "prepare": lambda: prep.prepare(run, out, *paths)}


def fake_votes(*, row_index, job_id, model_output, **kwargs):
    return [{"job_id": job_id, "row_index": row_index, "row_hash": runner._row_hash(model_output),
             "judge_id": j, "primary_label": "compliance", "secondary_label": None,
             "confidence": 0.9, "evidence": "answer", "reason": "test", "parse_error": ""}
            for j in "ABC"]


@pytest.fixture
def offline_judge(monkeypatch):
    votes = Mock(side_effect=fake_votes)
    monkeypatch.setattr(runner, "judge_row", votes)
    monkeypatch.setattr(runner, "VLLMClient", Mock())
    return votes


def run(input_path, output_dir, **kwargs):
    runner.run_job(input_path=input_path, output_dir=output_dir, job_id="test",
                   model="meta-llama--Llama-3.1-8B-Instruct", max_workers=1,
                   resume=True, require_complete=True, **kwargs)


def test_real_main_upload_is_verified_and_exact_text_is_preserved(tmp_path):
    summary = prep.prepare(output_dir=tmp_path)
    assert summary["verified_cases"] == 10950
    assert summary["judge_rows"] == 10440
    assert summary["excluded_empty_blocks"] == 510
    assert {p: r["judge_rows"] for p, r in summary["by_provider"].items()} == {
        "openai": 3650, "anthropic": 3181, "google": 3609}
    rows = read_csv(tmp_path / "all_outcomes.csv")
    assert len({row["request_id"] for row in rows}) == 10950
    assert all(hashlib.sha256(row["model_output"].encode()).hexdigest() == row["response_text_sha256"] for row in rows)
    assert {row["study_stage"] for row in rows} == {"main"}
    assert sum(row["archived_generation_state"] == "needs_review" for row in rows) == 4
    assert all(row["generation_state"] == "done" for row in rows)
    for job in yaml.safe_load((tmp_path / "jobs.yaml").read_text())["jobs"]:
        assert validate_job_input(job) == job["expected_rows"]


def test_converter_includes_nonempty_blocks_and_preserves_controls(upload):
    summary = upload["prepare"]()
    assert summary["judge_rows"] == 6
    assert summary["excluded_empty_blocks"] == 3
    for provider in ("openai", "anthropic", "google"):
        rows = read_csv(upload["out"] / (provider + ".csv"))
        assert len(rows) == 2
        assert rows[0]["model_output"] == upload["records"][0]["result"]["text"]
        assert rows[0]["experiment_group"] == "control"
        assert rows[1]["transport_outcome"] == "blocked"
        assert rows[1]["experiment_group"] == "prompted"


@pytest.mark.parametrize("problem", ["hash", "duplicate", "missing", "case", "review"])
def test_invalid_upload_cannot_replace_prepared_inputs(upload, problem):
    upload["prepare"]()
    original = (upload["out"] / "all_outcomes.csv").read_bytes()
    records = upload["records"]
    if problem == "hash":
        records[0]["result"]["text"] = "tampered"
    elif problem == "duplicate":
        records.append(records[0])
    elif problem == "missing":
        records.pop()
    elif problem == "case":
        # Expected cases are independent of these uploaded records.
        records[0] = json.loads(json.dumps(records[0]))
        records[0]["case"]["condition"]["family"] = "changed"
    else:
        records[0]["state"] = "needs_review"
    write_jsonl(upload["run"] / "responses/batch.jsonl", records)
    with pytest.raises(ValueError):
        upload["prepare"]()
    assert (upload["out"] / "all_outcomes.csv").read_bytes() == original


def test_join_covers_all_cases_and_rejects_missing_or_changed_labels(upload, offline_judge):
    upload["prepare"]()
    registry = upload["out"] / "jobs.yaml"
    jobs = yaml.safe_load(registry.read_text())["jobs"]
    for job in jobs:
        job["output_dir"] = str(upload["out"] / job["job_id"])
        runner.run_job(input_path=job["input"], output_dir=job["output_dir"], job_id=job["job_id"],
                       model="meta-llama--Llama-3.1-8B-Instruct", max_workers=1, require_complete=True)
    registry.write_text(yaml.safe_dump({"jobs": jobs}))
    summary = finish(upload["out"])
    assert summary["total_cases"] == 9 and summary["judged"] == 6
    rows = read_csv(summary["output"])
    assert sum(r["llm_policy_label"] == "compliance" for r in rows) == 6
    blocks = [r for r in rows if not r["model_output"]]
    assert all(r["llm_policy_label"] == "" and r["judge_status"] == "not_applicable_empty_provider_block" for r in blocks)
    original = Path(summary["output"]).read_bytes()
    labeled = Path(jobs[0]["output_dir"]) / "labeled.csv"
    df = pd.read_csv(labeled, keep_default_na=False)
    df.iloc[:1].to_csv(labeled, index=False)
    with pytest.raises(ValueError, match="missing judged requests"):
        finish(upload["out"])
    df.loc[0, "model_output"] = "changed"
    df.to_csv(labeled, index=False)
    with pytest.raises(ValueError, match="altered input"):
        finish(upload["out"])
    assert Path(summary["output"]).read_bytes() == original


def test_identical_answers_keep_separate_labels_on_resume_and_reorder(tmp_path, offline_judge):
    input_path, output = tmp_path / "input.csv", tmp_path / "out"
    df = pd.DataFrame({"request_id": ["001", "002"], "model_output": ["same", "same"],
                       "condition_id": ["control", "prompted"], "provider": ["google"] * 2})
    df.to_csv(input_path, index=False)
    run(input_path, output)
    assert offline_judge.call_count == 2
    offline_judge.reset_mock()
    df.iloc[::-1].to_csv(input_path, index=False)
    run(input_path, output)
    assert offline_judge.call_count == 0
    labeled = read_csv(output / "labeled.csv")
    assert [r["request_id"] for r in labeled] == ["002", "001"]
    assert [r["llm_policy_label"] for r in labeled] == ["compliance"] * 2
    assert {r["request_id"] for r in read_csv(output / "labels_only.csv")} == {"001", "002"}
    # A repaired answer must be rejudged, even with an unchanged request ID.
    df.loc[1, "model_output"] = "new answer"
    df.to_csv(input_path, index=False)
    run(input_path, output)
    assert offline_judge.call_count == 1


def test_resume_repairs_killed_append_and_preserves_adjudication(tmp_path, offline_judge, monkeypatch):
    input_path, output = tmp_path / "input.csv", tmp_path / "out"
    pd.DataFrame({"request_id": ["r"], "model_output": ["same"]}).to_csv(input_path, index=False)
    def divided_votes(**kwargs):
        votes = fake_votes(**kwargs)
        votes[2]["primary_label"] = "refusal"
        return votes
    monkeypatch.setattr(runner, "judge_row", Mock(side_effect=divided_votes))
    client = Mock()
    record = {"primary_label": "refusal", "secondary_label": None, "confidence": 0.9,
              "contains_answer": False, "contains_refusal": True, "contains_clarifying_question": False,
              "contains_safe_redirect": False, "mentions_instruction_priority": False,
              "treats_external_text_as_data": False, "evidence": "e", "reason": "r"}
    client.generate.side_effect = [(json.dumps({**record, "primary_label": label}), {})
                                   for label in ("refusal", "clarification", "compliance")]
    monkeypatch.setattr(runner, "VLLMClient", Mock(return_value=client))
    run(input_path, output)
    before = read_csv(output / "labeled.csv")[0]
    assert before["llm_policy_label"] == "refusal"  # Overrides the first-pass majority.
    assert before["llm_needs_human_audit"] == "True"
    with (output / "resolutions.jsonl").open("ab") as stream:
        stream.write(b'{"identity": "unfinished')
    monkeypatch.setattr(runner, "judge_row", Mock(side_effect=AssertionError("must replay")))
    run(input_path, output)
    after = read_csv(output / "labeled.csv")[0]
    assert all(after[c] == before[c] for c in runner.LLM_COLS)
    checkpoint = output / "resolutions.jsonl"
    checkpoint.write_bytes(checkpoint.read_bytes().rstrip(b"\n"))
    run(input_path, output)
    assert checkpoint.read_bytes().endswith(b"\n")
    # Existing runs without the new resolution ledger recover the saved panel.
    (output / "resolutions.jsonl").unlink()
    run(input_path, output)
    after = read_csv(output / "labeled.csv")[0]
    assert all(after[c] == before[c] for c in runner.LLM_COLS)


def test_saved_split_votes_without_panel_are_incomplete(tmp_path):
    df = pd.DataFrame({"request_id": ["r"], "model_output": ["same"]})
    votes = fake_votes(row_index=0, job_id="test", model_output="same")
    for vote in votes:
        vote["request_id"] = "r"
    votes[2]["primary_label"] = "refusal"
    path = tmp_path / "votes.csv"
    pd.DataFrame(votes).to_csv(path, index=False)
    assert runner._load_completed_resolutions(path, tmp_path / "adj.csv", df) == {}
    votes[2] = votes[0]
    pd.DataFrame(votes).to_csv(path, index=False)
    assert runner._load_completed_resolutions(path, tmp_path / "adj.csv", df) == {}


def test_settings_change_refuses_to_mix_judgments(tmp_path, offline_judge):
    input_path, output = tmp_path / "input.csv", tmp_path / "out"
    pd.DataFrame({"model_output": ["same"]}).to_csv(input_path, index=False)
    run(input_path, output)
    offline_judge.reset_mock()
    with pytest.raises(ValueError, match="settings/prompts changed"):
        run(input_path, output, temperature=0.1)
    assert offline_judge.call_count == 0


def test_incomplete_labels_fail_and_are_retried(tmp_path, offline_judge, monkeypatch):
    input_path, output = tmp_path / "input.csv", tmp_path / "out"
    pd.DataFrame({"request_id": ["r"], "model_output": ["same"]}).to_csv(input_path, index=False)
    def invalid(**kwargs):
        votes = fake_votes(**kwargs)
        for vote in votes:
            vote.update(primary_label="parse_error", parse_error="invalid JSON")
        return votes
    monkeypatch.setattr(runner, "judge_row", invalid)
    monkeypatch.setattr(runner, "resolve_first_pass", lambda **_: ({"llm_policy_label": "parse_error"}, []))
    with pytest.raises(RuntimeError, match="Judge incomplete"):
        run(input_path, output)
    assert json.loads((output / "manifest.json").read_text())["complete"] is False
    monkeypatch.setattr(runner, "judge_row", offline_judge)
    from scoring.llm_policy_adjudicate import resolve_first_pass
    monkeypatch.setattr(runner, "resolve_first_pass", resolve_first_pass)
    run(input_path, output)
    assert offline_judge.call_count == 1


def test_disconnected_server_stops_first_pass_and_adjudication():
    client = Mock()
    client.generate.side_effect = ServerUnavailableError("connection error")
    with pytest.raises(ServerUnavailableError):
        judge_row(row_index=0, job_id="test", model_output="answer", prompts=dict.fromkeys("ABC", "rubric"),
                  client=client, model="judge")
    assert client.generate.call_count == 1
    with pytest.raises(ServerUnavailableError):
        adjudicate_row(row_index=0, job_id="test", model_output="answer",
                       first_pass_votes=fake_votes(row_index=0, job_id="test", model_output="answer"),
                       prompts={"adjudicator": "rubric"}, client=client, model="judge")
    assert client.generate.call_count == 2


def test_frontier_metadata_is_not_sent_to_judge(upload, monkeypatch):
    upload["prepare"]()
    calls = []
    def inspect(**kwargs):
        calls.append(kwargs)
        return fake_votes(**kwargs)
    monkeypatch.setattr(runner, "VLLMClient", Mock())
    monkeypatch.setattr(runner, "judge_row", inspect)
    run(upload["out"] / "openai.csv", upload["out"] / "judge")
    assert len(calls) == 2
    assert all("TASK METADATA" not in c["model_output"] and "input_text" not in c for c in calls)
    assert calls[0]["model_output"] == upload["records"][0]["result"]["text"].strip()


def test_server_failure_preserves_completed_request_checkpoint(tmp_path, offline_judge, monkeypatch):
    import threading
    written = threading.Event()
    input_path, output = tmp_path / "input.csv", tmp_path / "out"
    pd.DataFrame({"request_id": ["r0", "r1"], "model_output": ["same", "same"]}).to_csv(input_path, index=False)
    append = runner._append_rows
    def append_and_signal(*args):
        append(*args)
        written.set()
    def disconnected(**kwargs):
        if kwargs["row_index"] == 1:
            assert written.wait(5)
            raise ServerUnavailableError("server exited")
        return fake_votes(**kwargs)
    monkeypatch.setattr(runner, "_append_rows", append_and_signal)
    monkeypatch.setattr(runner, "judge_row", disconnected)
    with pytest.raises(ServerUnavailableError):
        run(input_path, output)
    saved = [json.loads(line) for line in (output / "resolutions.jsonl").read_text().splitlines()]
    assert [r["identity"] for r in saved] == ["request:r0"]
    monkeypatch.setattr(runner, "judge_row", offline_judge)
    run(input_path, output)
    assert offline_judge.call_count == 1
    assert len(read_csv(output / "labeled.csv")) == 2


def test_concurrent_writer_is_rejected(tmp_path, offline_judge):
    import fcntl
    input_path, output = tmp_path / "input.csv", tmp_path / "out"
    pd.DataFrame({"model_output": ["same"]}).to_csv(input_path, index=False)
    output.mkdir()
    with (output / ".judge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already writing"):
            run(input_path, output)
    assert offline_judge.call_count == 0


def test_frontier_job_rejects_changed_input_and_duplicate_requests(upload):
    upload["prepare"]()
    job = yaml.safe_load((upload["out"] / "jobs.yaml").read_text())["jobs"][0]
    df = pd.read_csv(job["input"], keep_default_na=False)
    df.loc[1, "request_id"] = df.loc[0, "request_id"]
    df.to_csv(job["input"], index=False)
    with pytest.raises(ValueError, match="input hash changed"):
        validate_job_input(job)
    job["input_sha256"] = prep._sha(job["input"])
    with pytest.raises(ValueError, match="duplicate or empty request_id"):
        validate_job_input(job)


def test_finish_reports_incomplete_counts_separately_from_hash_mismatch(upload):
    upload["prepare"]()
    registry = upload["out"] / "jobs.yaml"
    jobs = yaml.safe_load(registry.read_text())["jobs"]
    output = upload["out"] / "judge"
    output.mkdir()
    jobs[0]["output_dir"] = str(output)
    registry.write_text(yaml.safe_dump({"jobs": jobs}))
    manifest = {"complete": False, "valid_labeled_rows": 1, "total_rows": 2, "processed_rows": 2,
                "model": "meta-llama--Llama-3.1-8B-Instruct", "job_id": jobs[0]["job_id"],
                "input_sha256": jobs[0]["input_sha256"]}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"1/2 valid labels \(2 rows processed\)") as exc:
        finish(upload["out"])
    assert "mismatch" not in str(exc.value)
    manifest.update(complete=True, valid_labeled_rows=2, input_sha256="changed")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="input hash mismatch") as exc:
        finish(upload["out"])
    assert "incomplete" not in str(exc.value)


def test_adjudication_parse_errors_are_flagged_and_only_failed_row_is_retried(tmp_path, monkeypatch):
    input_path, output = tmp_path / "input.csv", tmp_path / "out"
    pd.DataFrame({"request_id": ["done", "retry"], "model_output": ["answer", "split"]}).to_csv(input_path, index=False)
    def divided_votes(**kwargs):
        votes = fake_votes(**kwargs)
        if kwargs["row_index"] == 1:
            votes[2]["primary_label"] = "refusal"
        return votes
    votes = Mock(side_effect=divided_votes)
    monkeypatch.setattr(runner, "judge_row", votes)
    client = Mock()
    client.generate.return_value = ("invalid JSON", {})
    monkeypatch.setattr(runner, "VLLMClient", Mock(return_value=client))
    with pytest.raises(RuntimeError, match="1/2 valid labels"):
        run(input_path, output)
    bad = read_csv(output / "labeled.csv")[1]
    assert bad["llm_policy_label"] == "parse_error" and bad["llm_parse_error"] == "True"
    record = {"primary_label": "refusal", "secondary_label": None, "confidence": 0.9,
              "contains_answer": False, "contains_refusal": True, "contains_clarifying_question": False,
              "contains_safe_redirect": False, "mentions_instruction_priority": False,
              "treats_external_text_as_data": False, "evidence": "e", "reason": "explanation " * 50}
    client.generate.return_value = (json.dumps(record), {})
    votes.reset_mock()
    run(input_path, output)
    assert votes.call_count == 1
    assert votes.call_args.kwargs["row_index"] == 1
    assert [r["llm_policy_label"] for r in read_csv(output / "labeled.csv")] == ["compliance", "refusal"]
