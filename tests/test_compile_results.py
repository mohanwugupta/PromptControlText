import csv
import json
from pathlib import Path

import pytest

from scoring.compile_results import compile_sources


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def row(**extra):
    return {"benchmark": "XSTest", "item_id": "item", "prompt_family": "Answer-first",
            "clarity_level": "explicit", "prompt_variant": "v1", "input_text": "task",
            "model_output": 'An answer, with "quotes"\nand a newline.', "llm_policy_label": "compliance",
            "model_name": "model", **extra}


def source(path, study="open_models", **extra):
    return {"path": path, "job_id": path.stem, "expected_rows": 1, "analysis_study": study,
            "model_cohort": "new_open_models" if study == "open_models" else "frontier_models",
            "experiment_group": "prompted", **extra}


def test_compiler_retains_all_labels_metadata_controls_and_empty_blocks(tmp_path):
    old, new, frontier = [tmp_path / (name + ".csv") for name in ("old", "new", "frontier")]
    write_csv(old, [row(model_name="", refusal_score="0.0", compliance_score="1.0")])
    write_csv(new, [row(item_id="item2", llm_policy_label="parse_error", prompt_family="No-system-prompt")])
    write_csv(frontier, [row(request_id="r1", provider="google", experiment_group="control", transport_outcome="text"),
                         row(request_id="r2", provider="google", experiment_group="prompted", transport_outcome="blocked",
                             model_output="", llm_policy_label="", reference_json='{"answer": "gold"}')])
    output = tmp_path / "combined.csv"
    report = compile_sources([source(old, default_model="Qwen2.5-72B", model_cohort="original_open_models"),
                              source(new, experiment_group="control"),
                              source(frontier, "frontier_main", expected_rows=2)], output)
    rows = read_csv(output)
    assert len(rows) == report["total_rows"] == 4
    assert rows[0]["model_output"] == row()["model_output"]
    assert rows[0]["model_name"] == "Qwen2.5-72B"
    assert rows[0]["compliance_score"] == "1.0"
    assert rows[1]["llm_policy_label"] == "parse_error" and rows[1]["judge_status"] == "invalid_judgment"
    assert rows[1]["experiment_group"] == rows[2]["experiment_group"] == "control"
    assert rows[2]["source_run"] == "frontier_main_google" and rows[2]["score_source"] == "original_heuristic_scorers"
    assert rows[3]["judge_status"] == "not_applicable_empty_provider_block"
    assert rows[3]["llm_policy_label"] == rows[3]["refusal_score"] == ""
    assert rows[3]["reference_json"] == '{"answer": "gold"}'
    assert report["by_study"] == {"open_models": 2, "frontier_main": 2}
    assert json.loads(output.with_suffix(".summary.json").read_text())["total_rows"] == 4


@pytest.mark.parametrize("failure", ["duplicate", "count", "model", "frontier_label"])
def test_compiler_failure_preserves_previous_combined_csv(tmp_path, failure):
    path, output = tmp_path / "source.csv", tmp_path / "combined.csv"
    output.write_text("previous data\n")
    rows, config = [row()], source(path)
    if failure == "duplicate":
        rows.append(row())
        config["expected_rows"] = 2
    elif failure == "count":
        config["expected_rows"] = 2
    elif failure == "model":
        rows[0]["model_name"] = ""
    else:
        config = source(path, "frontier_main")
        rows[0].update(request_id="r", provider="openai", transport_outcome="text", llm_policy_label="parse_error")
    write_csv(path, rows)
    with pytest.raises(ValueError):
        compile_sources([config], output)
    assert output.read_text() == "previous data\n"
    assert not list(tmp_path.glob("combined.csv.*.tmp"))


def test_duplicate_sources_and_self_input_are_rejected(tmp_path):
    path = tmp_path / "source.csv"
    write_csv(path, [row()])
    with pytest.raises(ValueError):
        compile_sources([source(path), source(path)], tmp_path / "out.csv")
    with pytest.raises(ValueError):
        compile_sources([source(path)], path)
