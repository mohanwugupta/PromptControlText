import csv
from pathlib import Path

from scoring.llm_policy_run_jobs import combine_labeled_outputs, load_jobs


ROOT = Path(__file__).resolve().parents[1]


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
