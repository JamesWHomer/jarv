"""Resume keeps a live transcript until the saved conversation actually changes."""

import os
import threading
from types import SimpleNamespace

import pytest
from rich.text import Text

from conftest import make_console
from jarv import cli, commands, headsup, history, session_commands, session_store
from jarv.agent import AgentRunResult, SessionPersistence
from jarv.command_input import TextInput
from jarv.headsup import HeadsupApp


def exchange(prompt, answer, frame_id):
    return [
        {"role": "user", "content": prompt, "id": frame_id},
        {"role": "assistant", "content": answer},
    ]


@pytest.fixture
def saved_app(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(history, "_session_override", None)
    monkeypatch.setattr(history, "detect_terminal", lambda: ("test-terminal", "Test"))
    monkeypatch.setattr(cli, "_setup_nudge", lambda: None)
    monkeypatch.setattr(headsup, "enable_mouse_wheel_reporting", lambda: None)
    console, _ = make_console(width=100, height=24, force_terminal=True)
    for module in (commands, session_commands):
        monkeypatch.setattr(module, "console", console)
    monkeypatch.setattr(cli, "_console", lambda: console)
    monkeypatch.setattr(headsup, "terminal_size", lambda **_: (100, 24))
    with history.session_override("saved-chat"):
        context = history.prepare_session_context(mark_message=True)
        history.save_history(exchange("saved question", "saved answer", "saved"), context.history_file)
    history.set_terminal_session(context.session_id)

    def handle(command, rest, config, client, args, hint):
        return cli._handle_heads_up_slash_command(
            command, rest, config=config, client=client, args=args,
            unknown_help_hint=hint,
        )

    ready = threading.Event()
    ready.set()
    app = HeadsupApp(
        {"provider": "openai", "model": "test"}, object(), args=None,
        agent_loader=({"module": SimpleNamespace()}, ready),
        handle_slash=handle, maybe_command=lambda *_: None, render_console=console,
    )
    app._sync_initial_transcript_from_history()
    return app


def transcript(app):
    return "\n".join(line.plain for line in app._transcript_lines(100))


def assert_same_entries(app, before):
    assert len(app.entries) == len(before)
    assert all(current is original for current, original in zip(app.entries, before))


def install_saved_turn(app, monkeypatch, *, after_save=None):
    """Use the actual UI and persistence together without a provider request."""
    def run_agent(query, config, _client, *, ui, **_kwargs):
        ui.start_turn(query, config)
        ui.finish_assistant_message("local answer")
        # This transient detail must survive a no-op resume after the save.
        app.add_usage(Text("local turn usage detail"))
        persistence = SessionPersistence(incognito=False, on_history_saved=ui.history_saved)
        persistence.session_context = history.prepare_session_context(mark_message=True)
        path = persistence.session_context.history_file
        persistence.history = history.load_history(path)
        ui.history_loaded(path, persistence.history)
        persistence.history.extend(exchange(query, "local answer", "local"))
        persistence.save_turn()
        if after_save is not None:
            after_save()
        return AgentRunResult(text="local answer")

    monkeypatch.setattr(app.agent_import["module"], "run_agent", run_agent, raising=False)


def test_unchanged_resume_preserves_live_entries_and_does_not_write(saved_app, monkeypatch):
    app = saved_app
    app.add_usage(Text("transient live detail"))
    before = tuple(app.entries)
    paths = (history.SESSIONS_FILE, app.session_context.history_file)
    files_before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("Resuming the current chat must not write session state")

    monkeypatch.setattr(history, "save_sessions", unexpected_write)
    monkeypatch.setattr(session_commands, "set_terminal_session", unexpected_write)
    for _ in range(2):
        app._run_slash("/resume", [])
        assert_same_entries(app, before)
        assert "transient live detail" in transcript(app)
        assert "Already in the latest chat for this directory." in transcript(app)
        assert "Use /sessions to choose another." in transcript(app)
        assert app._prompt_history == ["saved question"]
        assert app.session_context.session_id == "saved-chat"

    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == files_before


@pytest.mark.parametrize("change", ["append", "same_size_edit"])
def test_external_change_refreshes_same_session_transcript_and_recall(saved_app, change):
    app = saved_app
    app.add_usage(Text("transient live detail"))
    before = tuple(app.entries)
    path = app.session_context.history_file
    original_stat = path.stat()
    if change == "append":
        updated = list(history.load_history(path)) + exchange("other question", "other answer", "other")
        expected_prompts = ["saved question", "other question"]
    else:
        # All substituted text has the same length, and restore the old mtime:
        # checking only a file's size/timestamp cannot detect this external edit.
        updated = exchange("other question", "other answer", "saved")
        expected_prompts = ["other question"]
    history.save_history(updated, path)
    if change == "same_size_edit":
        assert path.stat().st_size == original_stat.st_size
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        assert path.stat().st_mtime_ns == original_stat.st_mtime_ns

    app._run_slash("/resume", [])

    assert app.session_context.session_id == "saved-chat"
    assert all(entry is not before[0] for entry in app.entries)
    assert "other question" in transcript(app)
    assert "other answer" in transcript(app)
    assert "transient live detail" not in transcript(app)
    assert app._prompt_history == expected_prompts
    app.on_key("UP", 1)
    assert app.editor["buffer"] == "other question"
    refreshed = tuple(app.entries)
    app._run_slash("/resume", [])
    assert_same_entries(app, refreshed)
    assert app.editor["buffer"] == "other question"


def test_local_save_preserves_live_transcript_until_an_external_change(saved_app, monkeypatch):
    app = saved_app
    install_saved_turn(app, monkeypatch)
    app.on_key(TextInput("local question"), 1)
    app.on_key("ENTER", 1)
    assert "local question" in transcript(app)
    assert "local answer" in transcript(app)
    assert "local turn usage detail" in transcript(app)
    assert history.load_history(app.session_context.history_file)[-1]["content"] == "local answer"
    local_entries = tuple(app.entries)

    app._run_slash("/resume", [])

    assert_same_entries(app, local_entries)
    assert app._prompt_history == ["saved question", "local question"]
    path = app.session_context.history_file
    updated = list(history.load_history(path)) + exchange("external question", "external answer", "external")
    history.save_history(updated, path)
    app._run_slash("/resume", [])
    assert "external answer" in transcript(app)
    assert "local turn usage detail" not in transcript(app)
    assert app._prompt_history == ["saved question", "local question", "external question"]


def test_btw_checkout_preserves_visible_aside_until_external_change(saved_app, monkeypatch):
    app = saved_app
    install_saved_turn(app, monkeypatch)

    app._run_slash("/btw", ["aside question"])

    path = app.session_context.history_file
    assert [item["content"] for item in history.load_history(path) if item.get("role") == "user"] == ["saved question"]
    assert "aside question" in transcript(app)
    assert "local answer" in transcript(app)
    assert "Set aside" in transcript(app)
    aside_entries = tuple(app.entries)

    app._run_slash("/resume", [])

    assert_same_entries(app, aside_entries)
    assert "aside question" in transcript(app)
    assert "local turn usage detail" in transcript(app)
    updated = list(history.load_history(path)) + exchange("external question", "external answer", "external")
    history.save_history(updated, path)
    app._run_slash("/resume", [])
    assert "external answer" in transcript(app)
    assert "aside question" not in transcript(app)
    assert "local turn usage detail" not in transcript(app)
    assert app._prompt_history == ["saved question", "external question"]


@pytest.mark.parametrize("local_turn", ["normal", "btw"])
def test_external_messages_loaded_by_local_turn_still_refresh_on_resume(saved_app, monkeypatch, local_turn):
    app = saved_app
    install_saved_turn(app, monkeypatch)
    path = app.session_context.history_file
    updated = list(history.load_history(path)) + exchange("hidden question", "hidden answer", "hidden")
    history.save_history(updated, path)

    if local_turn == "normal":
        app.on_key(TextInput("local question"), 1)
        app.on_key("ENTER", 1)
        expected_prompts = ["saved question", "hidden question", "local question"]
    else:
        app._run_slash("/btw", ["aside question"])
        expected_prompts = ["saved question", "hidden question"]
        assert "Set aside" in transcript(app)
        assert "aside question" in transcript(app)

    # The local turn incorporated another terminal's messages on disk, but
    # those messages have not been rendered in this terminal yet.
    assert "hidden question" not in transcript(app)
    assert "local answer" in transcript(app)
    assert "local turn usage detail" in transcript(app)
    assert [item["content"] for item in history.load_history(path) if item.get("role") == "user"] == expected_prompts

    app._run_slash("/resume", [])

    assert "hidden question" in transcript(app)
    assert "hidden answer" in transcript(app)
    assert "local turn usage detail" not in transcript(app)
    assert app._prompt_history == expected_prompts
    if local_turn == "btw":
        assert "aside question" not in transcript(app)
        assert "local answer" not in transcript(app)
    else:
        assert "local question" in transcript(app)
        assert "local answer" in transcript(app)
    refreshed = tuple(app.entries)
    app._run_slash("/resume", [])
    assert_same_entries(app, refreshed)


def test_resume_without_directory_match_preserves_live_transcript_despite_external_change(saved_app, tmp_path, monkeypatch):
    app = saved_app
    app.add_usage(Text("transient live detail"))
    before = tuple(app.entries)
    path = app.session_context.history_file
    updated = list(history.load_history(path)) + exchange("external question", "external answer", "external")
    history.save_history(updated, path)
    elsewhere = tmp_path / "without-saved-chats"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    metadata_before = history.SESSIONS_FILE.read_bytes()

    app._run_slash("/resume", [])

    assert "No previous session found in this directory." in transcript(app)
    assert "Use /sessions to browse all saved sessions." in transcript(app)
    assert_same_entries(app, before)
    assert "transient live detail" in transcript(app)
    assert "external question" not in transcript(app)
    assert app.session_context.session_id == "saved-chat"
    assert app._prompt_history == ["saved question"]
    assert history.SESSIONS_FILE.read_bytes() == metadata_before


def test_new_launch_first_local_turn_resumes_without_rebuilding(saved_app, monkeypatch):
    # --new leaves the previous saved chat available, while the new app starts
    # from an empty conversation and records its first turn through normal UI.
    app = HeadsupApp(
        saved_app.config, saved_app.client, args=SimpleNamespace(new=True),
        agent_loader=({"module": SimpleNamespace()}, saved_app.agent_ready),
        handle_slash=saved_app.handle_slash, maybe_command=saved_app.maybe_command,
        render_console=saved_app.console,
    )
    app._sync_initial_transcript_from_history()
    assert app.session_context.session_id != saved_app.session_context.session_id
    assert "saved question" not in transcript(app)
    install_saved_turn(app, monkeypatch)
    app.on_key(TextInput("first question"), 1)
    app.on_key("ENTER", 1)
    assert "first question" in transcript(app)
    assert "local answer" in transcript(app)
    assert "local turn usage detail" in transcript(app)
    before = tuple(app.entries)
    session_id = app.session_context.session_id

    app._run_slash("/resume", [])

    assert app.session_context.session_id == session_id
    assert_same_entries(app, before)
    assert app._prompt_history == ["first question"]
    assert "Already in the latest chat for this directory." in transcript(app)
    assert "local turn usage detail" in transcript(app)


def test_resume_without_directory_match_ignores_external_terminal_rebinding(saved_app, tmp_path, monkeypatch):
    app = saved_app
    app.add_usage(Text("transient live detail"))
    before = tuple(app.entries)
    context_before = app.session_context
    usage_path_before = app.usage_path
    with history.session_override("different-chat"):
        other = history.prepare_session_context(mark_message=True)
        history.save_history(exchange("different question", "different answer", "different"), other.history_file)
    history.set_terminal_session(other.session_id)
    elsewhere = tmp_path / "without-saved-chats"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert history.prepare_session_context(persist_metadata=False).session_id == "different-chat"
    metadata_before = history.SESSIONS_FILE.read_bytes()

    app._run_slash("/resume", [])

    assert "No previous session found in this directory." in transcript(app)
    assert_same_entries(app, before)
    assert app.session_context == context_before
    assert app.usage_path == usage_path_before
    assert app._prompt_history == ["saved question"]
    assert "transient live detail" in transcript(app)
    assert "different question" not in transcript(app)
    assert history.SESSIONS_FILE.read_bytes() == metadata_before


def test_btw_checkout_keeps_intervening_external_edit_detectable(saved_app, monkeypatch):
    app = saved_app
    path = app.session_context.history_file

    def edit_earlier_exchange():
        updated = history.load_history(path)
        updated[0]["content"] = "edited question"
        updated[1]["content"] = "edited answer"
        history.save_history(updated, path)

    install_saved_turn(app, monkeypatch, after_save=edit_earlier_exchange)
    app._run_slash("/btw", ["aside question"])
    assert "Set aside" in transcript(app)
    assert "aside question" in transcript(app)
    assert "saved question" in transcript(app)
    assert "edited question" not in transcript(app)
    assert history.load_history(path)[0]["content"] == "edited question"

    app._run_slash("/resume", [])

    assert "edited question" in transcript(app)
    assert "edited answer" in transcript(app)
    assert "saved question" not in transcript(app)
    assert "aside question" not in transcript(app)
    assert app._prompt_history == ["edited question"]
    refreshed = tuple(app.entries)
    app._run_slash("/resume", [])
    assert_same_entries(app, refreshed)
