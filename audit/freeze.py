"""
audit/freeze.py
===============
Milestone 1 — Artifact freezer (PRD v4.1 Stage A).

Copies all required exploratory artifacts from a source directory into a
time-stamped versioned sub-directory under ``dest``, computes SHA-256 hashes
for every copied file, and writes a ``freeze_manifest.json`` alongside them.

No silent overwrites: each call creates a new versioned directory even when
one already exists for the same date (a counter suffix is appended).

Usage
-----
    from audit.freeze import freeze_artifacts
    freeze_artifacts(
        source_dir="artifacts/mining/2026-04-28",
        dest_dir="artifacts/frozen",
        required_files=REQUIRED_ARTIFACT_FILES,
        params={"n_clusters": 7, "registry_version": "v3"},
    )
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACT_FILES: List[str] = [
    "mining_table.csv",
    "exemplars_text_clustering.csv",
    "routing_sensitivity.csv",
    "enrichment_table.csv",
    "cluster_report.txt",
    "routing_sensitive_report.txt",
    "taxonomy_memo.txt",
    "manifest.json",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hash_file(path: Union[str, Path]) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def freeze_artifacts(
    source_dir: Union[str, Path],
    dest_dir: Union[str, Path],
    required_files: List[str],
    params: Optional[Dict] = None,
    label: Optional[str] = None,
) -> Path:
    """
    Copy *required_files* from *source_dir* into a new versioned subdirectory
    under *dest_dir*.

    Parameters
    ----------
    source_dir:
        Directory that contains the artifacts to freeze.
    dest_dir:
        Parent directory under which the versioned snapshot will be created.
    required_files:
        Names of files that **must** be present in *source_dir*.
        Raises ``FileNotFoundError`` if any are missing.
    params:
        Optional dict of run parameters to record in the manifest.
    label:
        Optional human-readable label for this snapshot (e.g. ``"v3_baseline"``).

    Returns
    -------
    Path
        The newly created versioned directory.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # 1. Validate that every required file is present in source.
    for name in required_files:
        candidate = source_dir / name
        if not candidate.exists():
            raise FileNotFoundError(
                f"Required artifact '{name}' not found in source dir '{source_dir}'."
            )

    # 2. Create a unique versioned sub-directory (date + counter to avoid overwrites).
    today = datetime.date.today().isoformat()
    base_name = f"{today}_{label}" if label else today
    versioned = dest_dir / base_name
    counter = 0
    while versioned.exists():
        counter += 1
        versioned = dest_dir / f"{base_name}_{counter:02d}"
    versioned.mkdir(parents=True, exist_ok=False)

    # 3. Copy files and compute hashes.
    file_entries = []
    for name in required_files:
        src_path = source_dir / name
        dst_path = versioned / name
        shutil.copy2(src_path, dst_path)
        file_entries.append(
            {
                "name": name,
                "sha256": hash_file(dst_path),
                "size_bytes": dst_path.stat().st_size,
            }
        )

    # 4. Write freeze manifest.
    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "source_dir": str(source_dir),
        "label": label,
        "params": params or {},
        "files": file_entries,
    }
    (versioned / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return versioned
