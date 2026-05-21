"""
analysis/llm_policy_compare.py

Compare rule-based labels against LLM judge labels.
Recompute policy distributions by family, clarity, and benchmark.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

VALID_LABELS = [
    "compliance", "refusal", "clarification",
    "hierarchy_preservation", "source_isolation", "safe_redirection",
]

RULE_COL = "classified_policy"
LLM_COL  = "llm_policy_label"


def _bar_chart(data: pd.DataFrame, x: str, hue: str, title: str, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_data = (
        data.groupby([x, hue]).size()
            .reset_index(name="count")
    )
    total = plot_data.groupby(x)["count"].transform("sum")
    plot_data["pct"] = plot_data["count"] / total
    pivot = plot_data.pivot(index=x, columns=hue, values="pct").fillna(0)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Proportion")
    ax.set_xlabel(x)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compare(
    combined_csv: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> None:
    combined_csv = pathlib.Path(combined_csv)
    output_dir   = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(combined_csv)

    # ── 1. Disagreement matrix ────────────────────────────────────────────
    if RULE_COL in df.columns and LLM_COL in df.columns:
        sub = df.dropna(subset=[RULE_COL, LLM_COL])
        dis_matrix = pd.crosstab(sub[RULE_COL], sub[LLM_COL])
        dis_matrix.to_csv(output_dir / "rule_vs_llm_disagreement_matrix.csv")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(dis_matrix, annot=True, fmt="d", ax=ax, cmap="Blues")
        ax.set_title("Rule label vs LLM label")
        ax.set_xlabel("LLM label")
        ax.set_ylabel("Rule label")
        plt.tight_layout()
        fig.savefig(output_dir / "rule_vs_llm_heatmap.png", dpi=150)
        plt.close(fig)

        agree_rate = (sub[RULE_COL] == sub[LLM_COL]).mean()
        print(f"Rule–LLM agreement: {agree_rate:.3f} ({len(sub)} rows)")

    # ── 2. Policy distribution by prompt_family ───────────────────────────
    for label_col, suffix in [(LLM_COL, "llm"), (RULE_COL, "rule")]:
        if label_col not in df.columns:
            continue
        for groupby_col in ("prompt_family", "clarity_level", "benchmark"):
            if groupby_col not in df.columns:
                continue
            sub = df.dropna(subset=[groupby_col, label_col])
            _bar_chart(
                sub, x=groupby_col, hue=label_col,
                title=f"Policy distribution ({suffix}) by {groupby_col}",
                out_path=output_dir / f"policy_dist_{suffix}_by_{groupby_col}.png",
            )

    # ── 3. Cross-tab family × clarity ────────────────────────────────────
    if "prompt_family" in df.columns and "clarity_level" in df.columns and LLM_COL in df.columns:
        ct = (
            df.dropna(subset=["prompt_family", "clarity_level", LLM_COL])
              .groupby(["prompt_family", "clarity_level", LLM_COL])
              .size()
              .reset_index(name="count")
        )
        ct.to_csv(output_dir / "family_clarity_llm_policy.csv", index=False)

    print(f"Comparison artifacts saved to {output_dir}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--combined-csv", required=True,
                   help="phase1_combined.csv with both classified_policy and llm_policy_label")
    p.add_argument("--output-dir", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    compare(args.combined_csv, args.output_dir)
