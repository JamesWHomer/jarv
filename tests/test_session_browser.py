import io
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from conftest import SnapshotLive, neutralize_tui_modes
from rich.console import Console

from jarv import session_browser


# Sessions in the order the browser sorts them (newest last_used_at first).
_ORDERED_SESSIONS = [
    ("parent-123456789abc", "2026-06-22T00:00:00Z", "hello"),
    ("second-23456789abcd", "2026-06-21T00:00:00Z", "hi again"),
    ("third-3456789abcde", "2026-06-20T00:00:00Z", "third one"),
    ("fourth-456789abcdef", "2026-06-19T00:00:00Z", "fourth one"),
]


def _extra_sessions(count):
    """The 2nd..Nth rows, for tests that need a range to select over."""
    return {
        sid: {
            "label": f"Session {sid}",
            "last_used_at": ts,
            "first_user_snippet": snippet,
        }
        for sid, ts, snippet in _ORDERED_SESSIONS[1:count]
    }


class TtyStdin:
    def isatty(self):
        return True


def _run_sessions_with_keys(monkeypatch, keys, extra_sessions=None):
    SnapshotLive.instances = []
    queued = deque(keys)
    loaded_sessions = []
    session_id = "parent-123456789abc"
    data = {
        "terminals": {"term-1": session_id},
        "sessions": {
            session_id: {
                "label": "Test session",
                "last_used_at": "2026-06-22T00:00:00Z",
                "first_user_snippet": "hello",
            }
        },
    }
    if extra_sessions:
        data["sessions"].update(extra_sessions)
    # Every session needs a history file for the archive/delete paths to engage;
    # the store itself is faked below, so the path only has to be distinct.
    for sid, meta in data["sessions"].items():
        meta.setdefault("history_file", f"history-{sid}.json")

    archived = []
    unarchived = []
    deleted = []

    def fake_archive(history_path):
        archived.append(str(history_path))
        return Path(f"archive/{Path(history_path).name}")

    def fake_unarchive(archived_path, sid):
        unarchived.append(sid)
        return Path(f"history-{sid}.json")

    def fake_delete(history_path):
        deleted.append(str(history_path))

    output = io.StringIO()
    test_console = Console(
        file=output,
        force_terminal=True,
        color_system=None,
        width=100,
        height=24,
    )

    neutralize_tui_modes(monkeypatch)
    monkeypatch.setattr(session_browser.sys, "stdin", TtyStdin())
    monkeypatch.setattr(session_browser, "console", test_console)
    monkeypatch.setattr(session_browser, "terminal_size", lambda *, console: (100, 24))
    monkeypatch.setattr(session_browser, "Live", SnapshotLive)
    monkeypatch.setattr(session_browser, "detect_terminal", lambda: ("term-1", "Terminal 1"))
    monkeypatch.setattr(session_browser, "load_sessions", lambda: data)
    monkeypatch.setattr(session_browser, "save_sessions", lambda _data: None)
    monkeypatch.setattr(session_browser, "set_terminal_session", loaded_sessions.append)
    monkeypatch.setattr(session_browser, "archive_session_files", fake_archive)
    monkeypatch.setattr(session_browser, "unarchive_session_files", fake_unarchive)
    monkeypatch.setattr(session_browser, "delete_session_files", fake_delete)

    def read_key_with_repeats(**_kwargs):
        if not queued:
            raise AssertionError("sessions loop requested an extra key")
        key = queued.popleft()
        if isinstance(key, tuple):
            return key
        return key, 1

    monkeypatch.setattr(session_browser, "_read_key_with_repeats", read_key_with_repeats)
    # The loop polls key availability before each read; keys are "available"
    # while the script still has some queued.
    monkeypatch.setattr(session_browser, "_key_available", lambda: bool(queued))

    session_browser.cmd_sessions([])
    assert not queued
    return SimpleNamespace(
        live=SnapshotLive.instances[-1],
        loaded=loaded_sessions,
        output=output.getvalue(),
        sessions=data["sessions"],
        terminals=data["terminals"],
        archived=archived,
        unarchived=unarchived,
        deleted=deleted,
    )


def _archived_ids(run):
    return sorted(sid for sid, meta in run.sessions.items() if meta.get("archived"))


def test_sessions_wheel_moves_selection(monkeypatch):
    # Raw MOUSE_WHEEL_* tokens (real mouse capture) fold onto selection
    # movement: wheel-down selects the next row, Enter loads it.
    run = _run_sessions_with_keys(
        monkeypatch,
        [("MOUSE_WHEEL_DOWN", 1), "ENTER"],
        extra_sessions=_extra_sessions(2),
    )

    assert run.loaded == ["second-23456789abcd"]
    assert "Loaded" in run.output


def test_sessions_delete_confirmation_esc_cancels_without_closing(monkeypatch):
    run = _run_sessions_with_keys(
        monkeypatch,
        ["d", "ESC", "ENTER"],
    )

    assert run.loaded == ["parent-123456789abc"]
    assert "Loaded" in run.output
    assert any("Delete parent-123456" in snapshot for snapshot in run.live.snapshots)


def test_sessions_shift_arrow_selects_range_and_archives_as_one_action(monkeypatch):
    # Shift+Down twice spans the top three rows; "a" archives the whole span,
    # and the batch reports as a single action rather than three.
    run = _run_sessions_with_keys(
        monkeypatch,
        ["SHIFT_DOWN", "SHIFT_DOWN", "a", "ESC", "ESC"],
        extra_sessions=_extra_sessions(4),
    )

    assert _archived_ids(run) == [
        "parent-123456789abc",
        "second-23456789abcd",
        "third-3456789abcde",
    ]
    assert len(run.archived) == 3
    assert any("3 selected" in snapshot for snapshot in run.live.snapshots)
    assert any("archived 3 sessions" in snapshot for snapshot in run.live.snapshots)


def test_sessions_shift_range_delete_names_the_count_and_undoes_together(monkeypatch):
    # The confirm prompt names the batch, both rows go, and one "u" brings the
    # whole batch back before the undo window can finalize the file removal.
    run = _run_sessions_with_keys(
        monkeypatch,
        ["SHIFT_DOWN", "d", "d", "u", "ESC", "ESC"],
        extra_sessions=_extra_sessions(4),
    )

    assert any(
        "Delete 2 sessions permanently?" in snapshot for snapshot in run.live.snapshots
    )
    assert set(run.sessions) == {
        "parent-123456789abc",
        "second-23456789abcd",
        "third-3456789abcde",
        "fourth-456789abcdef",
    }
    # Undo landed inside the window, so nothing was ever removed from disk.
    assert run.deleted == []
    assert any("restored 2 sessions" in snapshot for snapshot in run.live.snapshots)


def test_sessions_plain_arrow_ends_the_shift_range(monkeypatch):
    # An unmodified move drops the span, so "a" falls back to the cursor row.
    run = _run_sessions_with_keys(
        monkeypatch,
        ["SHIFT_DOWN", "DOWN", "a", "ESC"],
        extra_sessions=_extra_sessions(4),
    )

    assert _archived_ids(run) == ["third-3456789abcde"]


def test_sessions_view_switch_ends_the_shift_range(monkeypatch):
    # Tab re-filters the list; a span built against the old view must not carry
    # over onto rows the user can no longer see.
    run = _run_sessions_with_keys(
        monkeypatch,
        ["SHIFT_DOWN", "TAB", "a", "ESC"],
        extra_sessions=_extra_sessions(4),
    )

    assert _archived_ids(run) == ["second-23456789abcd"]


def test_sessions_enter_loads_only_the_cursor_row(monkeypatch):
    # Loading is inherently single-session: Enter takes the cursor row and
    # discards the span rather than acting on all of it.
    run = _run_sessions_with_keys(
        monkeypatch,
        ["SHIFT_DOWN", "SHIFT_DOWN", "ENTER"],
        extra_sessions=_extra_sessions(4),
    )

    assert run.loaded == ["third-3456789abcde"]
    assert _archived_ids(run) == []
    assert run.deleted == []
