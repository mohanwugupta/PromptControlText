"""
audit/tracker.py
================
Milestone 2 — Scratchpad and TODO tracker helpers (PRD v4.1 Milestone 2).

Provides lightweight functions for maintaining the project scratchpad and
TODO files programmatically, so the pipeline can log decisions, failed
approaches, and task completions without manual editing.

All functions are pure filesystem operations (no database).  Files are
created automatically if they do not yet exist.

Usage
-----
    from audit.tracker import log_decision, add_todo, mark_done

    log_decision(
        "scratchpad.md",
        title="Switched to KMeans",
        body="HDBSCAN produced >50% noise on the current data.",
        implication="Re-run clustering after data triples.",
    )

    add_todo("TODO.md", "Run pilot audit on 150 responses.")
    mark_done("TODO.md", "Run pilot audit on 150 responses.")
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_file(path: Path, default_content: str = "") -> None:
    """Create *path* (and its parents) if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(default_content, encoding="utf-8")


def _now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Scratchpad helpers
# ---------------------------------------------------------------------------

def append_scratchpad(path: Union[str, Path], text: str) -> None:
    """
    Append *text* to the scratchpad at *path*.

    The file is created with a default ``# Scratchpad`` header if it does not
    already exist.
    """
    path = Path(path)
    _ensure_file(path, "# Scratchpad\n\n")
    existing = path.read_text(encoding="utf-8")
    separator = "\n" if existing.endswith("\n") else "\n\n"
    path.write_text(existing + separator + text + "\n", encoding="utf-8")


def log_decision(
    path: Union[str, Path],
    title: str,
    body: str,
    implication: Optional[str] = None,
) -> None:
    """
    Append a structured decision entry to the scratchpad.

    The entry records:
    * timestamp
    * title
    * body
    * implication (optional)
    """
    lines = [
        f"## Decision: {title}",
        f"*{_now()}*",
        "",
        body,
    ]
    if implication:
        lines += ["", f"**Implication:** {implication}"]
    lines.append("")
    append_scratchpad(path, "\n".join(lines))


def log_failed_approach(
    path: Union[str, Path],
    title: str,
    body: str,
    implication: str,
) -> None:
    """
    Append a failed-approach entry to the scratchpad.

    Preserving failed approaches is a first-class scientific practice:
    it prevents the team from re-trying known dead ends.
    """
    lines = [
        f"## Failed approach: {title}",
        f"*{_now()}*",
        "",
        body,
        "",
        f"**Implication:** {implication}",
        "",
    ]
    append_scratchpad(path, "\n".join(lines))


# ---------------------------------------------------------------------------
# TODO tracker helpers
# ---------------------------------------------------------------------------

_DEFAULT_TODO = "# TODO\n\n## Pending\n\n## Completed\n\n"


def add_todo(
    path: Union[str, Path],
    item: str,
    section: str = "Pending",
) -> None:
    """
    Add ``- [ ] {item}`` under the *section* heading in the TODO file.

    The file (and a default ``## Pending`` / ``## Completed`` scaffold) is
    created automatically if it does not yet exist.
    """
    path = Path(path)
    _ensure_file(path, _DEFAULT_TODO)
    text = path.read_text(encoding="utf-8")
    entry = f"- [ ] {item}"
    # Insert after the section heading
    heading = f"## {section}"
    if heading not in text:
        # Append a new section at the end
        text = text.rstrip() + f"\n\n{heading}\n\n{entry}\n"
    else:
        text = text.replace(heading, f"{heading}\n{entry}", 1)
    path.write_text(text, encoding="utf-8")


def mark_done(path: Union[str, Path], item_text: str) -> None:
    """
    Replace ``- [ ] {item_text}`` with ``- [x] {item_text}`` in the file.

    The item is marked in-place; no text is moved.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    old = f"- [ ] {item_text}"
    new = f"- [x] {item_text}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def list_open_todos(path: Union[str, Path]) -> List[str]:
    """
    Return a list of open (unchecked) TODO items from *path*.

    Each element is the full item text (without the ``- [ ] `` prefix).
    Done items (``- [x]``) are excluded.
    """
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    open_items = []
    for line in text.splitlines():
        m = re.match(r"^- \[ \] (.+)$", line)
        if m:
            open_items.append(m.group(1))
    return open_items
