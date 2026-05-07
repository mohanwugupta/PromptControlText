"""
audit/dashboard.py
==================
Stage B — Human-audit dashboard (PRD v4.1 §12).

A Streamlit single-page app for labeling model responses with the coarse
policy taxonomy defined in PRD v4.1 §8.

Launch
------
    streamlit run audit/dashboard.py -- --audit-file artifacts/audit/audit_set_blinded.csv

Or with an existing labels file to resume:
    streamlit run audit/dashboard.py -- \\
        --audit-file  artifacts/audit/audit_set_blinded.csv \\
        --labels-file artifacts/audit/labels_in_progress.csv

Design notes
------------
* Blinding: ``prompt_family``, ``cluster_text``, ``cluster_hybrid``,
  ``disagreement_score`` are hidden by default and can be revealed only
  after the primary label is saved (§12.3).
* Autosave: labels are written to disk on every Save action.
* Export: CSV and JSON export from the sidebar.
* Keyboard shortcut hints are shown alongside each label button.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Guard: provide a clear error if streamlit is not installed when imported
# outside a Streamlit context (e.g. during unit tests).
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_LABELS = [
    "direct_answer_or_compliance",
    "clarification",
    "refusal",
    "safe_partial_help_or_redirection",
    "hierarchy_or_protocol_artifact",
    "tool_or_format_artifact",
    "mixed_or_unclear",
]

SECONDARY_FLAGS = [
    "answers_user_request",
    "asks_clarification",
    "contains_refusal",
    "contains_redirection",
    "contains_safety_rationale",
    "appears_benchmark_artifact",
    "unsafe_if_compliant",
    "safety_relevant",
]

BLINDED_COLS = [
    "prompt_family",
    "cluster_text",
    "cluster_hybrid",
    "disagreement_score",
]

_EXPORT_COLS = [
    "audit_id", "item_id", "response_text", "benchmark", "stratum",
    "primary_policy_label", "secondary_policy_label_optional",
    "confidence_1_to_5",
    "answers_user_request", "asks_clarification", "contains_refusal",
    "contains_redirection", "contains_safety_rationale",
    "appears_benchmark_artifact", "unsafe_if_compliant", "safety_relevant",
    "notes", "coder_id", "timestamp",
]


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_audit_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure all annotation columns exist with correct dtypes
    for col in SECONDARY_FLAGS:
        if col not in df.columns:
            df[col] = False
        else:
            df[col] = df[col].fillna(False).astype(bool)
    for col in ("primary_policy_label", "secondary_policy_label_optional",
                "notes", "coder_id", "timestamp"):
        if col not in df.columns:
            df[col] = ""
        else:
            # Cast to object (string) so label writes don't hit dtype mismatch
            df[col] = df[col].astype(object).fillna("")
    if "confidence_1_to_5" not in df.columns:
        df["confidence_1_to_5"] = 0
    else:
        df["confidence_1_to_5"] = pd.to_numeric(
            df["confidence_1_to_5"], errors="coerce"
        ).fillna(0).astype(int)
    return df


def _save_labels(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _find_next_unlabeled(df: pd.DataFrame, current: int) -> int:
    unlabeled = df[df["primary_policy_label"] == ""].index.tolist()
    after = [i for i in unlabeled if i > current]
    return after[0] if after else (unlabeled[0] if unlabeled else current)


# ---------------------------------------------------------------------------
# Main dashboard entry point
# ---------------------------------------------------------------------------

def run_dashboard(audit_file: str, labels_file: str, coder_id: str,
                  queue_file: Optional[str] = None) -> None:
    """Main Streamlit dashboard logic."""
    if not _STREAMLIT_AVAILABLE:
        raise ImportError("streamlit is required to run the dashboard. "
                          "Install it with: pip install streamlit")

    st.set_page_config(
        page_title="Policy-Routing Audit Dashboard",
        page_icon="🔍",
        layout="wide",
    )

    # ---------- Session state initialisation ----------
    if "df" not in st.session_state:
        full_df = _load_audit_file(audit_file)
        if queue_file and Path(queue_file).exists():
            q = pd.read_csv(queue_file)
            # Filter to only the audit_ids in the queue, preserving queue order
            ordered_ids = q["audit_id"].tolist() if "audit_id" in q.columns else []
            if ordered_ids:
                full_df = full_df[full_df["audit_id"].isin(ordered_ids)].copy()
                id_order = {aid: i for i, aid in enumerate(ordered_ids)}
                full_df["_queue_order"] = full_df["audit_id"].map(id_order)
                full_df = full_df.sort_values("_queue_order").drop(columns=["_queue_order"])
                full_df = full_df.reset_index(drop=True)
        st.session_state.df = full_df
        if labels_file and Path(labels_file).exists():
            saved = pd.read_csv(labels_file)
            # Merge saved labels back into the audit df
            label_cols = [c for c in saved.columns if c in _EXPORT_COLS]
            for _, row in saved.iterrows():
                idx = st.session_state.df.index[
                    st.session_state.df["audit_id"] == row["audit_id"]
                ]
                if len(idx):
                    for col in label_cols:
                        if col in st.session_state.df.columns:
                            st.session_state.df.at[idx[0], col] = row[col]

    if "current_idx" not in st.session_state:
        nxt = _find_next_unlabeled(st.session_state.df, -1)
        st.session_state.current_idx = max(0, nxt)
    if "reveal_metadata" not in st.session_state:
        st.session_state.reveal_metadata = False

    df = st.session_state.df
    total = len(df)
    labeled = int((df["primary_policy_label"] != "").sum())
    idx = st.session_state.current_idx
    row = df.iloc[idx]

    # ---------- Sidebar ----------
    with st.sidebar:
        st.title("🔍 Audit Dashboard")
        st.markdown(f"**Coder:** `{coder_id}`")
        st.markdown(f"**Progress:** {labeled} / {total} labeled")
        st.progress(labeled / total if total > 0 else 0)

        st.markdown("---")
        st.markdown("### Navigation")
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅ Prev", use_container_width=True):
                st.session_state.current_idx = max(0, idx - 1)
                st.session_state.reveal_metadata = False
                st.rerun()
        with col_next:
            if st.button("Next ➡", use_container_width=True):
                st.session_state.current_idx = min(total - 1, idx + 1)
                st.session_state.reveal_metadata = False
                st.rerun()
        if st.button("⏭ Jump to next unlabeled", use_container_width=True):
            st.session_state.current_idx = _find_next_unlabeled(df, idx)
            st.session_state.reveal_metadata = False
            st.rerun()

        st.markdown("---")
        jump = st.number_input("Jump to index:", min_value=0,
                               max_value=total - 1, value=idx, step=1)
        if st.button("Go"):
            st.session_state.current_idx = int(jump)
            st.session_state.reveal_metadata = False
            st.rerun()

        st.markdown("---")
        st.markdown("### Filters")
        show_low_conf = st.checkbox("Show only low-confidence (< 3)")
        show_unclear = st.checkbox("Show only mixed_or_unclear")

        st.markdown("---")
        st.markdown("### Export")
        export_df = df[_EXPORT_COLS] if all(c in df.columns for c in _EXPORT_COLS) else df
        st.download_button(
            "⬇ Download labels (CSV)",
            data=export_df.to_csv(index=False).encode(),
            file_name="audit_labels.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇ Download labels (JSON)",
            data=export_df.to_json(orient="records", indent=2).encode(),
            file_name="audit_labels.json",
            mime="application/json",
        )

    # ---------- Main panel ----------
    st.title(f"Response {idx + 1} / {total}")
    st.caption(
        f"**audit_id:** `{row.get('audit_id', '?')}` | "
        f"**item_id:** `{row.get('item_id', '?')}` | "
        f"**benchmark:** `{row.get('benchmark', '?')}` | "
        f"**stratum:** `{row.get('stratum', '?')}`"
    )

    # User request (input_text) — shown above the model response
    input_text = str(row.get("input_text", "")).strip()
    if input_text:
        with st.expander("📩 User request (input_text)", expanded=True):
            st.markdown(
                f"<div style='background:#f0f4ff;padding:12px 14px;"
                f"border-radius:6px;border-left:4px solid #4a7cdc;"
                f"font-size:0.95em;white-space:pre-wrap'>{input_text}</div>",
                unsafe_allow_html=True,
            )

    # Response text
    st.markdown("### Response")
    st.text_area(
        "Model output",
        value=str(row.get("response_text", row.get("model_output", ""))),
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

    # Reveal button (only enabled if primary label already saved)
    already_labeled = str(row.get("primary_policy_label", "")) != ""
    if st.button(
        "🔓 Reveal hidden metadata" if not st.session_state.reveal_metadata
        else "🔒 Hide metadata",
        disabled=not already_labeled,
        help="Available only after saving the primary label.",
    ):
        st.session_state.reveal_metadata = not st.session_state.reveal_metadata
        st.rerun()

    if st.session_state.reveal_metadata and already_labeled:
        with st.expander("Hidden metadata (unblinded)", expanded=True):
            for col in BLINDED_COLS:
                if col in df.columns:
                    st.markdown(f"**{col}:** `{row.get(col, 'N/A')}`")

    st.markdown("---")

    # ---------- Labeling form ----------
    with st.form("label_form"):
        st.markdown("### Primary policy label")
        primary = st.selectbox(
            "Select label:",
            options=[""] + POLICY_LABELS,
            index=(
                ([""] + POLICY_LABELS).index(str(row["primary_policy_label"]))
                if str(row["primary_policy_label"]) in POLICY_LABELS
                else 0
            ),
        )

        secondary = st.selectbox(
            "Secondary label (optional):",
            options=[""] + POLICY_LABELS,
            index=(
                ([""] + POLICY_LABELS).index(str(row.get("secondary_policy_label_optional", "")))
                if str(row.get("secondary_policy_label_optional", "")) in POLICY_LABELS
                else 0
            ),
        )

        st.markdown("### Confidence")
        confidence = st.slider(
            "Confidence (1=very unsure, 5=very sure):",
            min_value=1, max_value=5,
            value=int(row.get("confidence_1_to_5") or 1),
        )

        st.markdown("### Secondary flags")
        flag_cols = st.columns(4)
        flag_values = {}
        for i, flag in enumerate(SECONDARY_FLAGS):
            with flag_cols[i % 4]:
                flag_values[flag] = st.checkbox(
                    flag.replace("_", " "),
                    value=bool(row.get(flag, False)),
                )

        st.markdown("### Notes")
        notes = st.text_area("Free-text notes:", value=str(row.get("notes", "")), height=80)

        submitted = st.form_submit_button("💾 Save & continue", type="primary")

    if submitted:
        import datetime as _dt
        i = st.session_state.current_idx
        st.session_state.df.at[i, "primary_policy_label"] = primary
        st.session_state.df.at[i, "secondary_policy_label_optional"] = secondary
        st.session_state.df.at[i, "confidence_1_to_5"] = confidence
        st.session_state.df.at[i, "notes"] = notes
        st.session_state.df.at[i, "coder_id"] = coder_id
        st.session_state.df.at[i, "timestamp"] = _dt.datetime.utcnow().isoformat() + "Z"
        for flag, val in flag_values.items():
            st.session_state.df.at[i, flag] = val
        _save_labels(st.session_state.df, labels_file)
        # Advance to next unlabeled
        st.session_state.current_idx = _find_next_unlabeled(
            st.session_state.df, i
        )
        st.session_state.reveal_metadata = False
        st.success("Saved.")
        st.rerun()


# ---------------------------------------------------------------------------
# CLI entrypoint (``streamlit run audit/dashboard.py -- --audit-file ...``)
# ---------------------------------------------------------------------------

if __name__ == "__main__" and _STREAMLIT_AVAILABLE:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--audit-file", default="artifacts/audit/audit_set_blinded.csv")
    parser.add_argument("--labels-file", default="artifacts/audit/labels_in_progress.csv")
    parser.add_argument("--coder-id", default="coder_1")
    parser.add_argument("--queue-file", default=None,
                        help="Optional CSV of priority audit_ids to restrict the session to.")
    # Streamlit passes its own args before "--"; grab only ours
    try:
        args, _ = parser.parse_known_args(sys.argv[1:])
    except SystemExit:
        args = parser.parse_args([])

    run_dashboard(
        audit_file=args.audit_file,
        labels_file=args.labels_file,
        coder_id=args.coder_id,
        queue_file=args.queue_file,
    )
