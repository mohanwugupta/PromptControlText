"""
audit/build_audit_set.py
========================
Build the 200-item priority audit queue for the policy-classifier validation.

Sampling strategy
-----------------
We want rows that are maximally informative about whether the classifier is
correct, so we draw from four strata (target 200 total):

  A. Classifier / scorer disagree           — 80 items  (most informative)
  B. Rare classifier labels                 — 40 items  (hierarchy_defer,
                                                         classified_refusal,
                                                         safe_partial)
  C. Balanced family × label coverage      — 50 items  (≥1 per family × top-3
                                                         labels)
  D. Random sample of the remainder         — 30 items  (sanity-check)

Within each stratum items are shuffled so the labeler sees no obvious pattern.

Output
------
  artifacts/audit/audit_set_blinded.csv   — all cols needed by dashboard,
                                            prompt_family hidden behind
                                            BLINDED_COLS (dashboard reveals
                                            only after primary label saved)
  artifacts/audit/audit_queue_200.csv     — just the ordered audit_ids so the
                                            dashboard --queue-file arg can use it
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pandas as pd

from scoring.policy_classifier import classify_policy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PHASE1_CSV   = "artifacts/phase1_results.csv"
OUT_DIR      = Path("artifacts/audit")
BLINDED_CSV  = OUT_DIR / "audit_set_blinded.csv"
QUEUE_CSV    = OUT_DIR / "audit_queue_200.csv"

SEED = 42
N_TOTAL      = 200
N_DISAGREE   = 80
N_RARE       = 40
N_FAMILY     = 50
N_RANDOM     = 30

RARE_LABELS  = {"hierarchy_defer", "classified_refusal", "safe_partial"}

# Scorer label → normalised label for comparison with classifier output.
# The scorer uses the old v1 names; we map them to the 6-class taxonomy.
SCORER_NORM = {
    "direct_answer_or_compliance": "direct_answer",
    "direct_answer":               "direct_answer",
    "full_compliance":             "direct_answer",
    "refusal":                     "refusal",
    "clarification":               "clarification",
    "safe_partial":                "safe_partial",
    "safe_partial_help_or_redirection": "safe_partial",
    "stop_compliance":             "hierarchy_defer",
    "hierarchy_defer":             "hierarchy_defer",
    "classified_refusal":          "classified_refusal",
    "unsafe_continuation":         "direct_answer",   # compliance by another name
    "mixed_or_unclear":            "mixed_or_unclear",
}


def _stable_audit_id(row: pd.Series) -> str:
    """Deterministic audit ID from item content (stable across re-runs)."""
    key = f"{row.get('item_id','')}|{row.get('prompt_family','')}|{row.get('prompt_variant','')}|{row.get('benchmark','')}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def main() -> None:
    print(f"Loading {PHASE1_CSV} …")
    df = pd.read_csv(PHASE1_CSV)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 1. Apply classifier to every row
    # ------------------------------------------------------------------
    print("Running classifier on model_output …")
    df["classified_policy"] = df["model_output"].fillna("").apply(classify_policy)

    # ------------------------------------------------------------------
    # 2. Normalise the scorer label for comparison
    # ------------------------------------------------------------------
    df["scorer_norm"] = df["policy_label"].fillna("").map(SCORER_NORM).fillna("unknown")

    # ------------------------------------------------------------------
    # 3. Flag disagreements
    # ------------------------------------------------------------------
    df["disagrees"] = (
        (df["classified_policy"] != df["scorer_norm"]) &
        (df["scorer_norm"] != "unknown") &
        (df["scorer_norm"] != "mixed_or_unclear")
    )

    print(f"  Disagreements: {df['disagrees'].sum():,} / {len(df):,} "
          f"({df['disagrees'].mean():.1%})")
    print("  Classifier distribution:\n",
          df["classified_policy"].value_counts().to_string())
    print("  Disagreement by class:\n",
          df[df["disagrees"]]["classified_policy"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 4. Stratified sampling
    # ------------------------------------------------------------------
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=SEED).values  # shuffle index

    selected_ids: set[int] = set()

    def _add(sub: pd.DataFrame, n: int, label: str) -> None:
        cands = sub[~sub.index.isin(selected_ids)]
        take  = cands.sample(min(n, len(cands)), random_state=SEED)
        selected_ids.update(take.index.tolist())
        print(f"  Stratum {label}: requested {n}, added {len(take)}")

    # A — disagreements
    _add(df[df["disagrees"]], N_DISAGREE, "A-disagree")

    # B — rare labels (sample evenly across the three rare classes)
    per_rare = N_RARE // len(RARE_LABELS)
    for lbl in RARE_LABELS:
        _add(df[df["classified_policy"] == lbl], per_rare, f"B-rare/{lbl}")

    # C — family × label coverage (at least 1 per combination, up to budget)
    families = df["prompt_family"].dropna().unique()
    top_labels = df["classified_policy"].value_counts().head(4).index.tolist()
    fam_label_pairs = [(f, l) for f in families for l in top_labels]
    for fam, lbl in fam_label_pairs:
        if len(selected_ids) - (N_DISAGREE + N_RARE) >= N_FAMILY:
            break
        _add(df[(df["prompt_family"] == fam) & (df["classified_policy"] == lbl)], 1,
             f"C-{fam}/{lbl}")

    # D — random remainder
    _add(df, N_RANDOM, "D-random")

    # ------------------------------------------------------------------
    # 5. Build output dataframe
    # ------------------------------------------------------------------
    audit_df = df.loc[sorted(selected_ids)].copy()
    audit_df = audit_df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle order
    audit_df["audit_id"] = audit_df.apply(_stable_audit_id, axis=1)

    # Trim to dashboard-relevant columns (blinded columns kept but revealed
    # only after labeling — the dashboard enforces this in the UI)
    keep_cols = [
        "audit_id", "item_id", "benchmark", "domain",
        "input_text", "model_output",
        # blinded
        "prompt_family", "prompt_variant",
        # pre-computed signals (useful for post-hoc analysis)
        "classified_policy", "policy_label", "scorer_norm", "disagrees",
        "gold_label",
    ]
    keep_cols = [c for c in keep_cols if c in audit_df.columns]
    audit_df = audit_df[keep_cols].copy()

    # Rename model_output → response_text for dashboard compatibility
    if "model_output" in audit_df.columns and "response_text" not in audit_df.columns:
        audit_df = audit_df.rename(columns={"model_output": "response_text"})

    # Add stratum label for transparency
    def _stratum(row: pd.Series) -> str:
        if row.get("disagrees", False):
            return "A-disagree"
        if row.get("classified_policy", "") in RARE_LABELS:
            return "B-rare"
        return "C/D-other"
    audit_df["stratum"] = audit_df.apply(_stratum, axis=1)

    # Blank annotation columns (dashboard fills these in)
    for col in ("primary_policy_label", "secondary_policy_label_optional",
                "confidence_1_to_5", "notes", "coder_id", "timestamp"):
        audit_df[col] = ""

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(BLINDED_CSV, index=False)
    print(f"\n✅ Audit set saved → {BLINDED_CSV}  ({len(audit_df)} rows)")

    queue_df = audit_df[["audit_id"]].copy()
    queue_df.to_csv(QUEUE_CSV, index=False)
    print(f"✅ Queue file saved → {QUEUE_CSV}")

    # ------------------------------------------------------------------
    # 7. Summary for the labeler
    # ------------------------------------------------------------------
    print("\n── Audit set composition ──────────────────────────")
    print("By stratum:\n", audit_df["stratum"].value_counts().to_string())
    print("\nBy classified_policy:\n",
          audit_df["classified_policy"].value_counts().to_string())
    print("\nBy prompt_family:\n",
          audit_df["prompt_family"].value_counts().to_string())
    print("\nBy benchmark:\n",
          audit_df["benchmark"].value_counts().to_string())
    print("────────────────────────────────────────────────────")
    print("\nLaunch dashboard with:")
    print(f"  streamlit run audit/dashboard.py -- "
          f"--audit-file {BLINDED_CSV} "
          f"--queue-file {QUEUE_CSV}")


if __name__ == "__main__":
    main()
