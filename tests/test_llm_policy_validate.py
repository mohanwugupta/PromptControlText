"""
tests/test_llm_policy_validate.py

TDD tests for scoring/llm_policy_validate.py
RED → GREEN
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import pandas as pd
import pytest

from scoring.llm_policy_validate import validate, VALID_LABELS


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_labeled_csv(tmp: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    df = pd.DataFrame(rows)
    path = tmp / "labeled.csv"
    df.to_csv(path, index=False)
    return path


def _make_human_csv(tmp: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    df = pd.DataFrame(rows)
    path = tmp / "human.csv"
    df.to_csv(path, index=False)
    return path


def _perfect_data(n: int = 20):
    """Return (labeled_rows, human_rows) with perfect agreement across two labels.
    Uses two label classes so Cohen's κ is well-defined (single-class → NaN).
    """
    labels = ["compliance", "refusal"] * (n // 2) + ["compliance"] * (n % 2)
    labeled = [{"row_hash": f"h{i}", "llm_policy_label": labels[i]} for i in range(n)]
    human   = [{"row_hash": f"h{i}", "human_policy_label": labels[i]} for i in range(n)]
    return labeled, human


# ── basic functionality ───────────────────────────────────────────────────────

def test_validate_perfect_agreement_kappa_is_one():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled, human = _perfect_data()
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        result = validate(
            labeled_csv=lp, human_labels_csv=hp,
            output_dir=tmp / "out",
        )
        assert result["cohen_kappa"] == 1.0


def test_validate_returns_n_matched():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled, human = _perfect_data(10)
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        result = validate(labeled_csv=lp, human_labels_csv=hp, output_dir=tmp / "out")
        assert result["n_matched"] == 10


def test_validate_creates_confusion_matrix_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled, human = _perfect_data()
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        out = tmp / "out"
        validate(labeled_csv=lp, human_labels_csv=hp, output_dir=out)
        assert (out / "confusion_matrix.csv").exists()


def test_validate_creates_per_label_metrics_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled, human = _perfect_data()
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        out = tmp / "out"
        validate(labeled_csv=lp, human_labels_csv=hp, output_dir=out)
        assert (out / "per_label_metrics.csv").exists()


def test_validate_creates_summary_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled, human = _perfect_data()
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        out = tmp / "out"
        validate(labeled_csv=lp, human_labels_csv=hp, output_dir=out)
        summary_path = out / "validation_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert "cohen_kappa" in data


def test_validate_per_label_metrics_has_all_labels():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled, human = _perfect_data()
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        out = tmp / "out"
        validate(labeled_csv=lp, human_labels_csv=hp, output_dir=out)
        df = pd.read_csv(out / "per_label_metrics.csv")
        for label in VALID_LABELS:
            assert label in df["label"].values


def test_validate_inner_join_only_matched_rows():
    """Only rows whose row_hash appears in both CSVs contribute."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled = [
            {"row_hash": "h0", "llm_policy_label": "compliance"},
            {"row_hash": "h1", "llm_policy_label": "refusal"},  # no human label
        ]
        human = [{"row_hash": "h0", "human_policy_label": "compliance"}]
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        result = validate(labeled_csv=lp, human_labels_csv=hp, output_dir=tmp / "out")
        assert result["n_matched"] == 1


def test_validate_partial_disagreement_kappa_less_than_one():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled = [{"row_hash": f"h{i}", "llm_policy_label": "compliance"} for i in range(10)]
        human = (
            [{"row_hash": f"h{i}", "human_policy_label": "compliance"} for i in range(5)]
            + [{"row_hash": f"h{i}", "human_policy_label": "refusal"} for i in range(5, 10)]
        )
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        result = validate(labeled_csv=lp, human_labels_csv=hp, output_dir=tmp / "out")
        assert result["cohen_kappa"] < 1.0


def test_validate_summary_has_all_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled, human = _perfect_data()
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        result = validate(labeled_csv=lp, human_labels_csv=hp, output_dir=tmp / "out")
        required = {"n_matched", "cohen_kappa", "overall_accuracy", "macro_f1", "weighted_f1"}
        assert required <= result.keys()


def test_validate_accuracy_one_on_perfect_agreement():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        labeled, human = _perfect_data(30)
        lp = _make_labeled_csv(tmp, labeled)
        hp = _make_human_csv(tmp, human)
        result = validate(labeled_csv=lp, human_labels_csv=hp, output_dir=tmp / "out")
        assert result["overall_accuracy"] == 1.0
