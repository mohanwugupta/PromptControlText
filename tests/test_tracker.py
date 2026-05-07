"""tests/test_tracker.py — Milestone 2: Scratchpad / TODO tracker (PRD v4.1)."""
from pathlib import Path
import pytest
from audit.tracker import (
    append_scratchpad,
    log_decision,
    log_failed_approach,
    add_todo,
    mark_done,
    list_open_todos,
)


# ---------------------------------------------------------------------------
# Scratchpad tests
# ---------------------------------------------------------------------------

def test_scratchpad_created_if_missing(tmp_path):
    p = tmp_path / "new_scratchpad.md"
    assert not p.exists()
    append_scratchpad(p, "First entry.")
    assert p.exists()
    assert "First entry." in p.read_text()


def test_append_scratchpad_preserves_existing_text(tmp_path):
    p = tmp_path / "scratchpad.md"
    p.write_text("# Existing\n\nOld content.\n")
    append_scratchpad(p, "New entry.")
    text = p.read_text()
    assert "Old content." in text
    assert "New entry." in text


def test_log_decision_contains_required_fields(tmp_path):
    p = tmp_path / "scratchpad.md"
    log_decision(p, title="Chose KMeans", body="KMeans was chosen because of determinism.")
    text = p.read_text()
    assert "Chose KMeans" in text
    assert "KMeans was chosen because of determinism." in text
    # Must include a timestamp
    assert "202" in text  # year prefix


def test_log_decision_with_implication(tmp_path):
    p = tmp_path / "scratchpad.md"
    log_decision(
        p,
        title="Registry v3",
        body="Extended to 8 families.",
        implication="Re-run all experiments.",
    )
    text = p.read_text()
    assert "Re-run all experiments." in text


def test_log_failed_approach_contains_implication(tmp_path):
    p = tmp_path / "scratchpad.md"
    log_failed_approach(
        p,
        title="HDBSCAN trial",
        body="HDBSCAN produced >50% noise points.",
        implication="Stick with KMeans until data grows.",
    )
    text = p.read_text()
    assert "HDBSCAN trial" in text
    assert "HDBSCAN produced >50% noise points." in text
    assert "Stick with KMeans" in text


# ---------------------------------------------------------------------------
# TODO tracker tests
# ---------------------------------------------------------------------------

def test_todo_created_if_missing(tmp_path):
    p = tmp_path / "new_TODO.md"
    assert not p.exists()
    add_todo(p, "First task.")
    assert p.exists()
    assert "First task." in p.read_text()


def test_add_todo_adds_checkbox_item(tmp_path):
    p = tmp_path / "TODO.md"
    p.write_text("# TODO\n\n## Pending\n\n")
    add_todo(p, "Build audit dashboard.")
    text = p.read_text()
    assert "- [ ] Build audit dashboard." in text


def test_mark_done_moves_item_to_done(tmp_path):
    p = tmp_path / "TODO.md"
    p.write_text("# TODO\n\n## Pending\n\n- [ ] Build audit dashboard.\n\n## Completed\n\n")
    mark_done(p, "Build audit dashboard.")
    text = p.read_text()
    # Should be marked done
    assert "- [x] Build audit dashboard." in text


def test_list_open_todos_excludes_done_items(tmp_path):
    p = tmp_path / "TODO.md"
    p.write_text(
        "# TODO\n\n## Pending\n\n"
        "- [ ] Task A.\n"
        "- [ ] Task B.\n\n"
        "## Completed\n\n"
        "- [x] Task C.\n"
    )
    open_todos = list_open_todos(p)
    assert "Task A." in " ".join(open_todos)
    assert "Task B." in " ".join(open_todos)
    assert "Task C." not in " ".join(open_todos)


def test_list_open_todos_empty_when_none(tmp_path):
    p = tmp_path / "TODO.md"
    p.write_text("# TODO\n\n## Completed\n\n- [x] Done thing.\n")
    assert list_open_todos(p) == []
