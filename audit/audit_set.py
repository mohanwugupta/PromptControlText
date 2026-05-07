"""
audit/audit_set.py
==================
Milestone 3 — Audit-set builder (PRD v4.1 Stage B / §13).

Constructs a stratified audit set from the mining-pipeline outputs.
Rows are deduplicated across strata so the same (item_id, prompt_family)
pair never appears twice.

Strata
------
1. ``routing_sensitive``  — complete item panels for the top routing-sensitive items.
2. ``random_panel``       — random item panels (all families per item).
3. ``safety_boundary``    — HarmBench / XSTest / IHEval-conflict items.
4. ``ambiguous_benign``   — items from ambiguous or under-specified domains.
5. ``artifact_protocol``  — rows flagged with ``malformed_output != ""``.
6. ``cluster_exemplar``   — n_exemplar_per_cluster rows per text cluster.

The returned DataFrame has:
* all annotation columns pre-filled with empty / zero defaults (ready for the
  human auditor to fill in).
* a unique ``audit_id`` (``<stratum_prefix>_<counter>``) per row.
* all source metadata present for the unblinded export.

``blind_audit_set()`` drops the columns listed in ``BLINDED_COLUMNS`` to
produce a blinded view for first-pass labeling.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PRD v4.1 §12.4 export schema — annotation columns
# ---------------------------------------------------------------------------

ANNOTATION_COLUMNS = [
    "primary_policy_label",
    "secondary_policy_label_optional",
    "confidence_1_to_5",
    "answers_user_request",
    "asks_clarification",
    "contains_refusal",
    "contains_redirection",
    "contains_safety_rationale",
    "appears_benchmark_artifact",
    "unsafe_if_compliant",
    "safety_relevant",
    "notes",
    "coder_id",
    "timestamp",
]

# Columns hidden from annotators during first-pass labeling (§12.3)
BLINDED_COLUMNS = [
    "prompt_family",
    "cluster_text",
    "cluster_hybrid",
    "disagreement_score",
]

# Columns that must be in mining_table for the builder to work
_REQUIRED_SOURCE_COLS = [
    "item_id",
    "prompt_family",
    "benchmark",
    "model_output",
]

# Domains considered "ambiguous benign" for Stratum 4
_AMBIGUOUS_DOMAINS = {
    "ambiguous", "unclear", "malformed", "low_context",
    "ambiguous_intent", "single_word",
}

# Benchmarks / domains used to identify safety-boundary rows for Stratum 3
_SAFETY_BENCHMARKS = {"HarmBench", "XSTest"}
_SAFETY_DOMAINS = {
    "harmful", "unsafe", "safe_sensitive", "sensitive",
    "roleplay", "jailbreak",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_annotation_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Add empty annotation columns in-place if they are missing."""
    df = df.copy()
    bool_flags = [
        "answers_user_request", "asks_clarification", "contains_refusal",
        "contains_redirection", "contains_safety_rationale",
        "appears_benchmark_artifact", "unsafe_if_compliant", "safety_relevant",
    ]
    df["primary_policy_label"] = ""
    df["secondary_policy_label_optional"] = ""
    df["confidence_1_to_5"] = 0
    for col in bool_flags:
        df[col] = False
    df["notes"] = ""
    df["coder_id"] = ""
    df["timestamp"] = ""
    return df


def _assign_audit_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add ``audit_id`` column using ``<prefix>_<zero-padded counter>``."""
    df = df.copy()
    df["audit_id"] = [f"{prefix}_{i:05d}" for i in range(len(df))]
    return df


def _deduplicate(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Return rows in *new* whose (item_id, prompt_family) pair is not in *base*."""
    if base.empty:
        return new
    seen = set(zip(base["item_id"], base["prompt_family"]))
    mask = ~new.apply(lambda r: (r["item_id"], r["prompt_family"]) in seen, axis=1)
    return new[mask]


# ---------------------------------------------------------------------------
# Stratum builders
# ---------------------------------------------------------------------------

def _stratum_routing_sensitive(
    mining_table: pd.DataFrame,
    routing_sensitive: pd.DataFrame,
    n_routing: int,
) -> pd.DataFrame:
    """Complete item panels for the top-n routing-sensitive items."""
    top_items = (
        routing_sensitive
        .sort_values("disagreement_score", ascending=False)
        .head(n_routing)["item_id"]
        .tolist()
    )
    rows = mining_table[mining_table["item_id"].isin(top_items)].copy()
    rows["stratum"] = "routing_sensitive"
    return rows


def _stratum_random_panel(
    mining_table: pd.DataFrame,
    n_random: int,
    seed: int,
    exclude_ids: set,
) -> pd.DataFrame:
    """Random item panels — all families per sampled item."""
    eligible = mining_table[~mining_table["item_id"].isin(exclude_ids)]
    unique_items = eligible["item_id"].unique()
    rng = np.random.default_rng(seed)
    n = min(n_random, len(unique_items))
    sampled = rng.choice(unique_items, size=n, replace=False)
    rows = mining_table[mining_table["item_id"].isin(sampled)].copy()
    rows["stratum"] = "random_panel"
    return rows


def _stratum_safety_boundary(
    mining_table: pd.DataFrame,
    n_safety: int,
    seed: int,
    exclude_ids: set,
) -> pd.DataFrame:
    """HarmBench / XSTest / IHEval-conflict items."""
    mask = (
        mining_table["benchmark"].isin(_SAFETY_BENCHMARKS)
        | mining_table["domain"].isin(_SAFETY_DOMAINS)
    ) & ~mining_table["item_id"].isin(exclude_ids)
    eligible = mining_table[mask]
    unique_items = eligible["item_id"].unique()
    rng = np.random.default_rng(seed + 1)
    n = min(n_safety, len(unique_items))
    sampled = rng.choice(unique_items, size=n, replace=False)
    rows = mining_table[mining_table["item_id"].isin(sampled)
                        & mask].copy()
    rows["stratum"] = "safety_boundary"
    return rows


def _stratum_ambiguous_benign(
    mining_table: pd.DataFrame,
    n_ambiguous: int,
    seed: int,
    exclude_ids: set,
) -> pd.DataFrame:
    """Ambiguous / under-specified items."""
    mask = (
        mining_table["domain"].str.lower().isin(_AMBIGUOUS_DOMAINS)
        | mining_table["gold_label"].str.lower().isin({"ambiguous", "unclear"})
    ) & ~mining_table["item_id"].isin(exclude_ids)
    eligible = mining_table[mask]
    if eligible.empty:
        # Fall back to random rows if no ambiguous domain exists in the data
        eligible = mining_table[~mining_table["item_id"].isin(exclude_ids)]
    unique_items = eligible["item_id"].unique()
    rng = np.random.default_rng(seed + 2)
    n = min(n_ambiguous, len(unique_items))
    sampled = rng.choice(unique_items, size=n, replace=False) if n > 0 else []
    rows = mining_table[mining_table["item_id"].isin(sampled)].copy()
    rows["stratum"] = "ambiguous_benign"
    return rows


def _stratum_artifact_protocol(
    mining_table: pd.DataFrame,
    n_artifact: int,
    seed: int,
    exclude_ids: set,
) -> pd.DataFrame:
    """Rows flagged with malformed_output (tool_call_token / garbled_unicode)."""
    if "malformed_output" not in mining_table.columns:
        return pd.DataFrame(columns=mining_table.columns.tolist() + ["stratum"])
    mask = (
        mining_table["malformed_output"].notna()
        & (mining_table["malformed_output"] != "")
        & ~mining_table["item_id"].isin(exclude_ids)
    )
    eligible = mining_table[mask]
    rng = np.random.default_rng(seed + 3)
    n = min(n_artifact, len(eligible))
    rows = eligible.sample(n=n, random_state=int(rng.integers(0, 2**31))).copy()
    rows["stratum"] = "artifact_protocol"
    return rows


def _stratum_cluster_exemplar(
    mining_table: pd.DataFrame,
    n_per_cluster: int,
    seed: int,
    exclude_ids: set,
) -> pd.DataFrame:
    """N rows per text cluster, avoiding already-selected items."""
    if "cluster_text" not in mining_table.columns or n_per_cluster == 0:
        return pd.DataFrame(columns=mining_table.columns.tolist() + ["stratum"])
    eligible = mining_table[~mining_table["item_id"].isin(exclude_ids)]
    rng = np.random.default_rng(seed + 4)
    chunks = []
    for cluster_id, group in eligible.groupby("cluster_text"):
        n = min(n_per_cluster, len(group))
        chunks.append(
            group.sample(n=n, random_state=int(rng.integers(0, 2**31)))
        )
    if not chunks:
        return pd.DataFrame(columns=mining_table.columns.tolist() + ["stratum"])
    rows = pd.concat(chunks, ignore_index=True).copy()
    rows["stratum"] = "cluster_exemplar"
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_audit_set(
    mining_table: pd.DataFrame,
    routing_sensitive: pd.DataFrame,
    *,
    n_routing: int = 20,
    n_random: int = 20,
    n_safety: int = 20,
    n_ambiguous: int = 10,
    n_artifact: int = 10,
    n_exemplar_per_cluster: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a stratified audit set from the mining pipeline outputs.

    Parameters
    ----------
    mining_table:
        Full mining table (one row per model output).
    routing_sensitive:
        DataFrame with at least ``item_id`` and ``disagreement_score`` columns.
    n_routing, n_random, n_safety, n_ambiguous, n_artifact:
        Target *item* counts per stratum (all families included per item).
    n_exemplar_per_cluster:
        Target row count per text cluster for Stratum 6.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Audit set with all annotation columns pre-filled to empty / False / 0,
        ready for the human auditor.
    """
    for col in _REQUIRED_SOURCE_COLS:
        if col not in mining_table.columns:
            raise ValueError(f"mining_table missing required column: '{col}'")

    # Rename model_output → response_text in the audit set
    mt = mining_table.copy()
    mt["response_text"] = mt["model_output"]

    strata_frames = []
    seen_pairs: set = set()

    def _add(stratum_df: pd.DataFrame) -> None:
        if stratum_df.empty:
            return
        deduped = _deduplicate(
            pd.concat(strata_frames, ignore_index=True) if strata_frames else pd.DataFrame(),
            stratum_df,
        )
        strata_frames.append(deduped)

    # Build strata in priority order
    _add(_stratum_routing_sensitive(mt, routing_sensitive, n_routing))
    seen_ids = set(pd.concat(strata_frames)["item_id"]) if strata_frames else set()

    _add(_stratum_random_panel(mt, n_random, seed, seen_ids))
    seen_ids = set(pd.concat(strata_frames)["item_id"]) if strata_frames else set()

    _add(_stratum_safety_boundary(mt, n_safety, seed, seen_ids))
    seen_ids = set(pd.concat(strata_frames)["item_id"]) if strata_frames else set()

    _add(_stratum_ambiguous_benign(mt, n_ambiguous, seed, seen_ids))
    seen_ids = set(pd.concat(strata_frames)["item_id"]) if strata_frames else set()

    _add(_stratum_artifact_protocol(mt, n_artifact, seed, seen_ids))
    seen_ids = set(pd.concat(strata_frames)["item_id"]) if strata_frames else set()

    _add(_stratum_cluster_exemplar(mt, n_exemplar_per_cluster, seed, seen_ids))

    if not strata_frames:
        raise ValueError("Audit set is empty — mining_table may be empty.")

    result = pd.concat(strata_frames, ignore_index=True)

    # Add annotation defaults
    result = _add_annotation_defaults(result)

    # Assign unique audit IDs
    result = result.reset_index(drop=True)
    result["audit_id"] = [f"AUD_{i:06d}" for i in range(len(result))]

    # Ensure response_text column exists
    if "response_text" not in result.columns:
        result["response_text"] = result.get("model_output", "")

    return result


def blind_audit_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* with all blinded columns removed.

    This is the view shown to annotators during first-pass labeling.
    """
    cols_to_drop = [c for c in BLINDED_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def save_audit_set(
    df: pd.DataFrame,
    path: str,
    blinded: bool = False,
) -> None:
    """
    Save the audit set to *path* as CSV.

    Parameters
    ----------
    df:
        Audit set DataFrame (as returned by ``build_audit_set``).
    path:
        Output file path (.csv).
    blinded:
        If True, drop blinded columns before saving.
    """
    import pathlib
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    export = blind_audit_set(df) if blinded else df
    export.to_csv(out, index=False)
