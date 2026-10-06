"""Directory-scoped resume through saved sessions and the real command dispatch."""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

from conftest import make_console
from jarv import cli, history, session_commands


@pytest.fixture
def saved_sessions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(history, "_session_override", None)
    monkeypatch.setattr(history, "detect_terminal", lambda: ("terminal", "Terminal"))
    monkeypatch.setattr(cli, "_setup_nudge", lambda: None)
    monkeypatch.setattr(cli, "_print_pending_results", lambda **_kwargs: None)
    console, output = make_console()
    monkeypatch.setattr(session_commands, "console", console)
    now = datetime(2026, 10, 5, tzinfo=timezone.utc)

    def save(session_id, *, directory=None):
        nonlocal now
        if directory is not None:
            monkeypatch.chdir(directory)
        now += timedelta(microseconds=1)
        monkeypatch.setattr(history, "utc_now", lambda: now)
        with history.session_override(session_id):
            context = history.prepare_session_context(mark_message=True)
            history.save_history([
                {"role": "user", "content": f"prompt for {session_id}"},
                {"role": "assistant", "content": f"reply for {session_id}"},
            ], context.history_file)
        return context

    return save, output


def test_resume_uses_latest_message_in_this_directory_across_terminals(saved_sessions, tmp_path, monkeypatch):
    save, output = saved_sessions
    older = save("older")
    latest = save("latest")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    save("unrelated", directory=elsewhere)
    monkeypatch.chdir(tmp_path)
    # Merely viewing an older session must not make it the most recent chat.
    with history.session_override(older.session_id):
        history.prepare_session_context()
    monkeypatch.setattr(history, "detect_terminal", lambda: ("new-terminal", "New terminal"))
    monkeypatch.setattr(sys, "argv", ["jarv", "/resume"])

    cli.main()

    context = history.prepare_session_context()
    assert context.session_id == latest.session_id
    assert history.load_history(context.history_file)[0]["content"] == "prompt for latest"
    assert history.load_sessions()["terminals"]["new-terminal"] == latest.session_id
    assert "Resumed session latest" in output.getvalue()


def test_session_keeps_separate_recency_for_each_directory(saved_sessions, tmp_path, monkeypatch):
    save, _ = saved_sessions
    save("travelling")
    save("project-latest")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    save("travelling", directory=elsewhere)

    assert history.latest_session_for_directory() == "travelling"
    monkeypatch.chdir(tmp_path)
    assert history.latest_session_for_directory() == "project-latest"


@pytest.mark.parametrize("unavailable", ["archived", "missing", "empty"])
def test_resume_skips_unavailable_newer_sessions(saved_sessions, unavailable):
    save, _ = saved_sessions
    save("available")
    newer = save("newer")
    if unavailable == "archived":
        data = history.load_sessions()
        data["sessions"]["newer"]["archived"] = True
        history.save_sessions(data)
    elif unavailable == "missing":
        newer.history_file.unlink()
    else:
        history.save_history([], newer.history_file)

    assert session_commands.cmd_resume() == 0
    assert history.prepare_session_context().session_id == "available"


def test_resume_after_new_does_not_create_an_empty_session(saved_sessions):
    save, _ = saved_sessions
    save("previous")
    history.forget_current_session()
    history.prepare_session_context()

    assert session_commands.cmd_resume() == 0
    assert history.prepare_session_context().session_id == "previous"
    assert set(history.load_sessions()["sessions"]) == {"previous"}


def test_no_match_preserves_binding_and_reports_failure(saved_sessions, tmp_path, monkeypatch):
    save, output = saved_sessions
    save("existing")
    history.set_terminal_session("existing")
    elsewhere = tmp_path / "empty-project"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    before = history.load_sessions()
    monkeypatch.setattr(sys, "argv", ["jarv", "/resume"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 1
    assert history.load_sessions() == before
    assert "No previous session found in this directory" in output.getvalue()


def test_legacy_session_becomes_eligible_only_after_sending_a_message(saved_sessions):
    save, _ = saved_sessions
    save("legacy")
    data = history.load_sessions()
    del data["sessions"]["legacy"]["directories"]
    history.save_sessions(data)
    history.set_terminal_session("legacy")

    history.prepare_session_context()
    assert history.latest_session_for_directory() is None
    history.prepare_session_context(mark_message=True)
    assert history.latest_session_for_directory() == "legacy"


def test_resume_respects_invocation_override(saved_sessions):
    save, _ = saved_sessions
    save("previous")
    history.set_terminal_session("terminal-session")

    with history.session_override("temporary-session"):
        assert session_commands.cmd_resume() == 0
        assert history.prepare_session_context().session_id == "previous"

    assert history.load_sessions()["terminals"]["terminal"] == "terminal-session"


def test_directory_keys_normalize_equivalent_paths(tmp_path):
    path = tmp_path / "project"
    path.mkdir()
    expected = history.session_directory(path)
    assert history.session_directory(path / ".." / "project") == expected
    if os.name == "nt":
        assert history.session_directory(str(path).swapcase()) == expected


def test_resume_rejects_arguments_before_switching(saved_sessions, monkeypatch):
    save, _ = saved_sessions
    save("previous")
    before = history.load_sessions()
    monkeypatch.setattr(sys, "argv", ["jarv", "/resume", "unexpected"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
    assert history.load_sessions() == before


def test_headsup_resume_replaces_transcript_and_usage_session(saved_sessions):
    import threading
    from types import SimpleNamespace

    from jarv.headsup import HeadsupApp
    from jarv.usage import usage_file_for

    save, _ = saved_sessions
    save("older")
    latest = save("latest")
    history.set_terminal_session("older")
    ready = threading.Event()
    ready.set()
    console, _ = make_console()

    def handle_slash(command, rest, config, client, *_args):
        assert cli._run_slash_command(command, rest)
        return config, client

    app = HeadsupApp(
        {"provider": "openai", "model": "test-model"}, object(),
        args=None,
        agent_loader=({"module": SimpleNamespace()}, ready),
        handle_slash=handle_slash, maybe_command=lambda *_args: None,
        render_console=console,
    )
    app._sync_initial_transcript_from_history()
    app._run_slash("/resume", [])

    rendered = "\n".join(line.plain for line in app._transcript_lines(100))
    assert "prompt for latest" in rendered
    assert "reply for latest" in rendered
    assert "prompt for older" not in rendered
    assert app.session_context.session_id == "latest"
    assert app.usage_path == usage_file_for(latest.history_file)
