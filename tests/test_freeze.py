"""tests/test_freeze.py — Milestone 1: Artifact freezer (PRD v4.1)."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from audit.freeze import freeze_artifacts, hash_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source_dir(tmp_path: Path) -> tuple[Path, list[str]]:
    """Create a minimal source directory with the required artifact files."""
    src = tmp_path / "source"
    src.mkdir()
    required = [
        "mining_table.csv",
        "exemplars_text_clustering.csv",
        "routing_sensitivity.csv",
        "enrichment_table.csv",
        "cluster_report.txt",
        "routing_sensitive_report.txt",
        "taxonomy_memo.txt",
        "manifest.json",
    ]
    for name in required:
        (src / name).write_text(f"content of {name}")
    return src, required


# ---------------------------------------------------------------------------
# Milestone 1 tests
# ---------------------------------------------------------------------------

def test_freeze_artifacts_creates_versioned_dir(tmp_path):
    src, required = _make_source_dir(tmp_path)
    dest = tmp_path / "frozen"
    freeze_artifacts(src, dest, required_files=required)
    # A single child directory should be created (version stamp)
    children = list(dest.iterdir())
    assert len(children) == 1
    assert children[0].is_dir()


def test_freeze_artifacts_copies_required_files(tmp_path):
    src, required = _make_source_dir(tmp_path)
    dest = tmp_path / "frozen"
    freeze_artifacts(src, dest, required_files=required)
    versioned = next(dest.iterdir())
    for name in required:
        assert (versioned / name).exists(), f"{name} missing from frozen dir"


def test_freeze_artifacts_writes_manifest(tmp_path):
    src, required = _make_source_dir(tmp_path)
    dest = tmp_path / "frozen"
    freeze_artifacts(src, dest, required_files=required, params={"n_clusters": 7})
    versioned = next(dest.iterdir())
    manifest_path = versioned / "freeze_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "timestamp" in manifest
    assert "params" in manifest
    assert manifest["params"]["n_clusters"] == 7
    assert "files" in manifest


def test_freeze_artifacts_hashes_files(tmp_path):
    src, required = _make_source_dir(tmp_path)
    dest = tmp_path / "frozen"
    freeze_artifacts(src, dest, required_files=required)
    versioned = next(dest.iterdir())
    manifest = json.loads((versioned / "freeze_manifest.json").read_text())
    # Every required file should have an entry with a sha256 hash
    file_entries = {e["name"]: e for e in manifest["files"]}
    for name in required:
        assert name in file_entries
        h = file_entries[name]["sha256"]
        assert len(h) == 64  # SHA-256 hex digest
        # Hash should match actual file content
        expected = hash_file(versioned / name)
        assert h == expected


def test_freeze_artifacts_fails_on_missing_required_file(tmp_path):
    src, required = _make_source_dir(tmp_path)
    dest = tmp_path / "frozen"
    # Remove one required file from source
    (src / "mining_table.csv").unlink()
    with pytest.raises(FileNotFoundError, match="mining_table.csv"):
        freeze_artifacts(src, dest, required_files=required)


def test_freeze_artifacts_no_silent_overwrite(tmp_path):
    """Calling freeze twice should create two distinct versioned dirs."""
    src, required = _make_source_dir(tmp_path)
    dest = tmp_path / "frozen"
    freeze_artifacts(src, dest, required_files=required)
    freeze_artifacts(src, dest, required_files=required)
    children = [c for c in dest.iterdir() if c.is_dir()]
    assert len(children) == 2


def test_hash_file_matches_sha256(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert hash_file(f) == expected
