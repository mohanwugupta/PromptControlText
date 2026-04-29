"""tests/test_mining_end_to_end.py — full pipeline integration tests."""

from pathlib import Path

import pandas as pd
import pytest

from mining.reports import run_full_pipeline

P1 = Path("tests/fixtures/mining_phase1.csv")
P2 = Path("tests/fixtures/mining_phase2.csv")


# ── full pipeline on fixture data ─────────────────────────────────────────────

def test_full_pipeline_runs(tmp_path):
    artifacts = run_full_pipeline(
        str(P1), str(P2),
        out_dir=str(tmp_path),
        n_clusters=4,
        random_state=42,
        run_viz=False,
    )
    assert isinstance(artifacts, dict)
    assert len(artifacts) > 0


def test_full_pipeline_outputs_all_expected_artifacts(tmp_path):
    artifacts = run_full_pipeline(
        str(P1), str(P2),
        out_dir=str(tmp_path),
        n_clusters=4,
        random_state=42,
        run_viz=False,
    )
    required = [
        "mining_table",
        "exemplars",
        "sensitivity",
        "fig_cluster_sizes",
        "fig_enrichment",
        "enrichment_table",
        "cluster_report",
        "routing_sensitive_report",
        "taxonomy_memo",
    ]
    for key in required:
        assert key in artifacts, f"Missing artifact: {key}"
        path = Path(artifacts[key])
        assert path.exists(), f"Artifact file missing: {path}"
        assert path.stat().st_size > 0, f"Artifact file empty: {path}"


def test_mining_table_has_cluster_columns(tmp_path):
    artifacts = run_full_pipeline(
        str(P1), str(P2),
        out_dir=str(tmp_path),
        n_clusters=3,
        random_state=42,
    )
    df = pd.read_csv(artifacts["mining_table"])
    assert "cluster_text" in df.columns
    assert "cluster_hybrid" in df.columns


def test_mining_table_row_count_matches_inputs(tmp_path):
    p1 = pd.read_csv(P1)
    p2 = pd.read_csv(P2)
    artifacts = run_full_pipeline(
        str(P1), str(P2),
        out_dir=str(tmp_path),
        n_clusters=3,
        random_state=42,
    )
    df = pd.read_csv(artifacts["mining_table"])
    assert len(df) == len(p1) + len(p2)


def test_sensitivity_csv_has_required_columns(tmp_path):
    artifacts = run_full_pipeline(
        str(P1), str(P2),
        out_dir=str(tmp_path),
        n_clusters=3,
        random_state=42,
    )
    sens = pd.read_csv(artifacts["sensitivity"])
    for col in ["item_id", "disagreement_score", "n_families", "cluster_entropy"]:
        assert col in sens.columns


def test_manifest_json_written(tmp_path):
    """A manifest.json should be written listing all artifact paths."""
    import json
    run_full_pipeline(
        str(P1), str(P2),
        out_dir=str(tmp_path),
        n_clusters=3,
        random_state=42,
    )
    import datetime
    dated = tmp_path / datetime.date.today().isoformat()
    manifest = dated / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert "mining_table" in data
