import csv
from pathlib import Path

import pytest

from scoring.llm_policy_run_jobs import combine_labeled_outputs, load_jobs, validate_job_input


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def judge_input(tmp_path):
    fields = ["benchmark", "item_id", "prompt_family", "clarity_level", "prompt_variant", "model_output"]
    rows = [dict(zip(fields, ["XSTest", f"x_{i}", "Answer-first", "explicit", "v1", "Line 1\nLine 2"]))
            for i in range(2)]
    path = tmp_path / "input.csv"

    def write(records):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    write(rows)
    return {"job_id": "new-model", "input": str(path), "expected_rows": 2}, rows, write


def test_complete_judge_input_counts_multiline_responses(judge_input):
    job, _, _ = judge_input
    assert validate_job_input(job) == 2


def test_partial_judge_input_rejected(judge_input):
    job, rows, write = judge_input
    write(rows[:1])
    with pytest.raises(ValueError, match="1/2 responses"):
        validate_job_input(job)


def test_duplicate_judge_conditions_rejected_despite_correct_row_count(judge_input):
    job, rows, write = judge_input
    write([rows[0], rows[0]])
    with pytest.raises(ValueError, match="duplicate generation condition"):
        validate_job_input(job)


def test_empty_judge_outputs_rejected(judge_input):
    job, rows, write = judge_input
    rows[0]["model_output"] = " \n "
    write(rows)
    with pytest.raises(ValueError, match="1 empty outputs"):
        validate_job_input(job)


def test_original_jobs_do_not_require_new_model_row_count():
    assert validate_job_input({"job_id": "original", "input": "unused.csv"}) is None


def test_job_registry_contains_exact_paper_runs():
    jobs = load_jobs(ROOT / "configs" / "llm_policy_jobs.yaml")
    assert [job["job_id"] for job in jobs] == [
        "phase1_qwen25_72b",
        "phase1_llama31_8b",
        "phase1_llama33_70b",
        "phase1_qwen2_7b",
        "control_qwen25_72b",
        "control_llama31_8b",
        "control_llama33_70b",
        "control_qwen2_7b",
    ]


def test_combine_labeled_outputs_adds_run_metadata(tmp_path):
    jobs = []
    for job_id, group, value, fields in [
        ("phase1_model", "prompted", "a", ["item_id", "model_output"]),
        ("control_model", "control", "b", ["item_id", "control_score"]),
    ]:
        output_dir = tmp_path / job_id
        output_dir.mkdir()
        with (output_dir / "labeled.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            row = {"item_id": value}
            row[fields[1]] = "line 1\nline 2" if group == "prompted" else "1"
            writer.writerow(row)
        jobs.append(
            {
                "job_id": job_id,
                "output_dir": str(output_dir),
                "experiment_group": group,
            }
        )

    combined_path = tmp_path / "combined.csv"
    rows_written = combine_labeled_outputs(jobs, combined_path)

    with combined_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows_written == 2
    assert [row["experiment_group"] for row in rows] == ["prompted", "control"]
    assert [row["source_run"] for row in rows] == ["phase1_model", "control_model"]
    assert rows[0]["model_output"] == "line 1\nline 2"
    assert rows[0]["control_score"] == ""
    assert rows[1]["model_output"] == ""
    assert rows[1]["control_score"] == "1"
