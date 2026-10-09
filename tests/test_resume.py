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
    assert "Resumed chat prompt for latest (latest)" in output.getvalue()


def test_repeated_resume_of_current_chat_does_not_write_or_select_an_older_chat(saved_sessions, monkeypatch):
    save, output = saved_sessions
    older = save("older")
    latest = save("latest")
    history.set_terminal_session("latest")
    paths = (history.SESSIONS_FILE, older.history_file, latest.history_file)
    before = {path: path.read_bytes() for path in paths}

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("Resuming the current chat must not write session state")

    monkeypatch.setattr(history, "save_sessions", unexpected_write)
    monkeypatch.setattr(session_commands, "set_terminal_session", unexpected_write)
    monkeypatch.setattr(sys, "argv", ["jarv", "/resume"])

    cli.main()
    cli.main()

    assert history.load_sessions()["terminals"]["terminal"] == "latest"
    assert {path: path.read_bytes() for path in paths} == before
    assert output.getvalue().count("Already in the latest chat for this directory.") == 2
    assert output.getvalue().count("Use /sessions to choose another.") == 2
    assert "Resumed chat" not in output.getvalue()


def test_resume_current_invocation_override_leaves_terminal_and_metadata_unchanged(saved_sessions, monkeypatch):
    save, output = saved_sessions
    save("latest")
    history.set_terminal_session("terminal-session")
    before = history.SESSIONS_FILE.read_bytes()

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("Resuming the current invocation must not change its binding")

    monkeypatch.setattr(history, "save_sessions", unexpected_write)
    monkeypatch.setattr(session_commands, "set_terminal_session", unexpected_write)

    with history.session_override("latest"):
        assert session_commands.cmd_resume() == 0
        assert history.prepare_session_context(persist_metadata=False).session_id == "latest"

    assert history.SESSIONS_FILE.read_bytes() == before
    assert history.load_sessions()["terminals"]["terminal"] == "terminal-session"
    assert "Already in the latest chat for this directory." in output.getvalue()


def test_resume_switch_shows_saved_title_and_short_id(saved_sessions):
    save, output = saved_sessions
    session_id = "windows-terminal-0123456789ab"
    save(session_id)
    data = history.load_sessions()
    data["sessions"][session_id]["title"] = "  Review\n  the launch plan  "
    history.save_sessions(data)

    assert session_commands.cmd_resume() == 0

    assert "Resumed chat Review the launch plan (windows-terminal-012345)" in output.getvalue()
    assert f"prompt for {session_id}" not in output.getvalue()


@pytest.mark.parametrize("source", ["saved_title", "first_prompt"])
def test_resume_title_is_literal_terminal_safe_and_bounded(saved_sessions, source):
    save, output = saved_sessions
    session = save("session-0123456789ab")
    title = "[bold]Plan[/bold]\n\x1b[31m\x07 " + "界" * 100 + " hidden ending"
    if source == "saved_title":
        data = history.load_sessions()
        data["sessions"][session.session_id]["title"] = title
        history.save_sessions(data)
    else:
        history.save_history([{"role": "user", "content": title}], session.history_file)

    assert session_commands.cmd_resume() == 0

    rendered = output.getvalue()
    assert "[bold]Plan[/bold] \\x1b[31m\\x07" in rendered
    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "…" in rendered
    assert "hidden ending" not in rendered
    assert "(session-012345)" in rendered


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
