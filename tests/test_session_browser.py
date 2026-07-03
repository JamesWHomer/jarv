import io
from collections import deque
from contextlib import contextmanager

from conftest import SnapshotLive
from rich.console import Console

from jarv import session_browser


class TtyStdin:
    def isatty(self):
        return True


@contextmanager
def noop_context(*_args, **_kwargs):
    yield


def _run_sessions_with_keys(monkeypatch, keys):
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
    output = io.StringIO()
    test_console = Console(
        file=output,
        force_terminal=True,
        color_system=None,
        width=100,
        height=24,
    )

    monkeypatch.setattr(session_browser.sys, "stdin", TtyStdin())
    monkeypatch.setattr(session_browser, "console", test_console)
    monkeypatch.setattr(session_browser, "terminal_size", lambda *, console: (100, 24))
    monkeypatch.setattr(session_browser, "Live", SnapshotLive)
    monkeypatch.setattr(session_browser, "mouse_capture", noop_context)
    monkeypatch.setattr(session_browser, "detect_terminal", lambda: ("term-1", "Terminal 1"))
    monkeypatch.setattr(session_browser, "load_sessions", lambda: data)
    monkeypatch.setattr(session_browser, "save_sessions", lambda _data: None)
    monkeypatch.setattr(session_browser, "set_terminal_session", loaded_sessions.append)

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
    return SnapshotLive.instances[-1], loaded_sessions, output.getvalue()


def test_sessions_delete_confirmation_esc_cancels_without_closing(monkeypatch):
    live, loaded_sessions, output = _run_sessions_with_keys(
        monkeypatch,
        ["d", "ESC", "ENTER"],
    )

    assert loaded_sessions == ["parent-123456789abc"]
    assert "Loaded" in output
    assert any("Delete parent-123456" in snapshot for snapshot in live.snapshots)
