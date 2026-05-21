"""
scoring/llm_policy_validate.py

Compare LLM judge labels against a human audit CSV.
Computes Cohen's κ, precision, recall, F1, and confusion matrix.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

VALID_LABELS = [
    "compliance", "refusal", "clarification",
    "hierarchy_preservation", "source_isolation", "safe_redirection",
]


def validate(
    *,
    labeled_csv: str | pathlib.Path,
    human_labels_csv: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    llm_label_col: str = "llm_policy_label",
    human_label_col: str = "human_policy_label",
    join_on: str = "row_hash",
) -> Dict:
    labeled = pd.read_csv(labeled_csv)
    human = pd.read_csv(human_labels_csv)

    merged = labeled.merge(human[[join_on, human_label_col]], on=join_on, how="inner")
    merged = merged.dropna(subset=[llm_label_col, human_label_col])

    y_llm   = merged[llm_label_col].tolist()
    y_human = merged[human_label_col].tolist()

    kappa   = cohen_kappa_score(y_human, y_llm)
    report  = classification_report(
        y_human, y_llm, labels=VALID_LABELS, output_dict=True, zero_division=0
    )
    cm      = confusion_matrix(y_human, y_llm, labels=VALID_LABELS)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=VALID_LABELS, columns=VALID_LABELS)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    # Per-label metrics
    rows = []
    for label in VALID_LABELS:
        m = report.get(label, {})
        rows.append({
            "label": label,
            "precision": round(m.get("precision", 0), 4),
            "recall":    round(m.get("recall", 0), 4),
            "f1":        round(m.get("f1-score", 0), 4),
            "support":   m.get("support", 0),
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "per_label_metrics.csv", index=False)

    summary = {
        "n_matched": len(merged),
        "cohen_kappa": round(kappa, 4),
        "overall_accuracy": round(report.get("accuracy", 0), 4),
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 4),
        "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0), 4),
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"n={len(merged)}  κ={kappa:.4f}  accuracy={summary['overall_accuracy']:.4f}")
    print(metrics_df.to_string(index=False))
    return summary


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--labeled-csv", required=True)
    p.add_argument("--human-labels-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--llm-label-col", default="llm_policy_label")
    p.add_argument("--human-label-col", default="human_policy_label")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    validate(
        labeled_csv=args.labeled_csv,
        human_labels_csv=args.human_labels_csv,
        output_dir=args.output_dir,
        llm_label_col=args.llm_label_col,
        human_label_col=args.human_label_col,
    )
