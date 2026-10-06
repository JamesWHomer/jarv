"""Command feedback must reach the rendered heads-up screen, not just a buffer."""

import threading
import time
from types import SimpleNamespace

import pytest
from rich.live import Live
from rich.text import Text

from conftest import make_console, wait_for
from headsup_harness import HeadsupHarness
from jarv import cli, commands, headsup, history, session_commands, session_store, undo_commands
from jarv.command_input import TextInput
from jarv.headsup import HeadsupApp


@pytest.fixture
def command_app(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(history, "_session_override", None)
    monkeypatch.setattr(history, "detect_terminal", lambda: ("test-terminal", "Test"))
    monkeypatch.setattr(cli, "_setup_nudge", lambda: None)
    monkeypatch.setattr(headsup, "enable_mouse_wheel_reporting", lambda: None)
    console, stream = make_console(width=80, height=24, force_terminal=True)
    for module in (commands, session_commands, undo_commands):
        monkeypatch.setattr(module, "console", console)
    monkeypatch.setattr(cli, "_console", lambda: console)
    ready = threading.Event()
    ready.set()

    def handle(command, rest, config, client, args, hint):
        return cli._handle_heads_up_slash_command(
            command, rest, config=config, client=client, args=args,
            unknown_help_hint=hint,
        )

    app = HeadsupApp(
        {"provider": "openai", "model": "test"}, object(), args=None,
        agent_loader=({"module": SimpleNamespace()}, ready),
        handle_slash=handle, maybe_command=lambda *_: None, render_console=console,
    )
    monkeypatch.setattr(headsup, "terminal_size", lambda **_: (console.width, console.height))
    return app, console, stream


def screen_text(app, console):
    with console.capture() as capture:
        console.print(app.render())
    return Text.from_ansi(capture.get()).plain


@pytest.mark.parametrize(("command", "rest", "message"), [
    ("/resume", [], "No previous session found in this directory."),
    ("/new", [], "Already on a new session."),
    ("/history", [], "No history yet."),
    ("/undo", [], "Nothing to undo."),
    ("/redo", [], "Nothing to redo."),
    ("/setup", ["invalid"], "Unknown setup step 'invalid'."),
    ("/unknown", [], "Unknown command: /unknown"),
    ("/new", ["unexpected"], "/new does not accept arguments."),
    ("/resume", ["unexpected"], "/resume does not accept arguments."),
    ("/update", ["unexpected"], "/update does not accept arguments."),
])
def test_empty_session_displays_command_feedback(command_app, command, rest, message):
    app, console, _ = command_app
    app._foreground_input_active = True
    original_session = app.session_context.session_id
    app._run_slash(command, rest)

    assert message in screen_text(app, console)
    assert app._idle_animation_active()
    assert "█" in screen_text(app, console)
    assert app._outro_started_at == 0
    assert app._prompt_notice is None
    assert app.session_context.session_id == original_session
    assert history.load_history(app.session_context.history_file) == []


def test_resume_missing_session_remains_visible_while_typing_in_real_loop(command_app, monkeypatch):
    with HeadsupHarness(width=80, height=24) as harness:
        monkeypatch.setattr(session_commands, "console", harness.app.console)

        def handle(command, rest, config, client, *_):
            cli._run_slash_command(command, rest)
            return config, client

        harness.app.handle_slash = handle
        harness.feed_text("/resume")
        harness.feed_key("enter")
        assert wait_for(lambda: "Use /sessions" in harness.plain_frame)
        assert "No previous session found in this directory." in harness.plain_frame
        harness.feed_text("next message")
        assert wait_for(lambda: harness.prompt_buffer == "next message")
        assert "Use /sessions" in harness.plain_frame


def test_session_switch_keeps_all_result_lines_in_transcript(command_app):
    app, console, _ = command_app
    with history.session_override("saved-session"):
        context = history.prepare_session_context(mark_message=True)
        history.save_history([{"role": "user", "content": "saved conversation"}], context.history_file)
    app.add_user_message("old visible conversation")

    app._run_slash("/resume", [])
    app.on_key(TextInput("draft"), 1)
    rendered = screen_text(app, console)
    assert "saved conversation" in rendered
    assert "Resumed session saved-session" in rendered
    assert "old visible conversation" not in rendered
    assert app._prompt_notice is None
    assert len(history.load_history(context.history_file)) == 1


def test_archive_shows_both_result_lines_after_clearing_history(command_app):
    app, console, _ = command_app
    context = history.prepare_session_context(mark_message=True)
    history.save_history([{"role": "user", "content": "conversation to archive"}], context.history_file)
    app._sync_transcript_from_history()
    app._run_slash("/archive", [])

    rendered = screen_text(app, console)
    assert "Session archived to" in rendered
    assert "New session starts on your next message." in rendered
    assert "conversation to archive" not in rendered
    assert app.session_context.session_id != context.session_id
    assert history.load_history(app.session_context.history_file) == []


@pytest.mark.parametrize("restricted", ["incognito", "busy"])
def test_restricted_command_is_visible_over_welcome_screen(command_app, restricted):
    app, console, _ = command_app
    app.incognito = restricted == "incognito"
    app._agent_busy = restricted == "busy"
    app._run_slash("/resume", [])
    assert "/resume is unavailable" in screen_text(app, console)


def test_feedback_reflows_and_reveals_itself_when_scrolled_up(command_app):
    app, console, _ = command_app
    for number in range(40):
        app.add_user_message(f"old message {number}")
    app.scroll_offset = 30
    console.width = 30
    message = "This command result is long enough to wrap in a narrow terminal."

    def handle(_command, _rest, config, client, *_args):
        console.print(Text(message, style="green"))
        console.print("The second line must also survive.")
        return config, client

    app.handle_slash = handle
    app._run_slash("/set", ["model", "test"])
    assert app.scroll_offset == 0
    console.width = 100
    rendered = screen_text(app, console)
    assert message in rendered
    assert "The second line must also survive." in rendered
    assert "\x1b" not in "".join(line.plain for line in app._transcript_lines(100))


def test_nested_live_frames_are_not_captured_but_its_messages_are(command_app):
    app, console, stream = command_app

    def handle(_command, _rest, config, client, *_args):
        console.print("Before opening the view.")
        with Live(
            Text("NESTED SCREEN CONTENT"), console=console, screen=True,
            auto_refresh=False, redirect_stdout=False, redirect_stderr=False,
        ) as nested:
            nested.refresh()
        console.print("After closing the view.")
        return config, client

    app.handle_slash = handle
    with Live(
        get_renderable=app.render, console=console, screen=True,
        auto_refresh=False, redirect_stdout=False, redirect_stderr=False,
    ) as parent:
        app.live = parent
        app._run_slash("/history", [])
        assert console._render_hooks == [parent]
    app.live = None
    assert "NESTED SCREEN CONTENT" in stream.getvalue()
    rendered = screen_text(app, console)
    assert "Before opening the view." in rendered
    assert "After closing the view." in rendered
    assert "NESTED SCREEN CONTENT" not in rendered


@pytest.mark.parametrize("command", ["/set", "/settings"])
def test_command_exception_preserves_feedback_and_restores_console(command_app, command):
    app, console, _ = command_app

    def handle(*_args):
        console.print("Starting command.")
        raise ValueError("example failure")

    app.handle_slash = handle
    app._run_slash(command, [])
    rendered = screen_text(app, console)
    assert "Starting command." in rendered
    assert f"{command} failed: example failure" in rendered
    assert not console._render_hooks
    assert app._refresh_suspended == 0


def test_session_refresh_failure_still_shows_command_output(command_app, monkeypatch):
    app, console, _ = command_app

    def broken_refresh():
        raise OSError("cannot read session")

    monkeypatch.setattr(app, "_refresh_session_context", broken_refresh)
    app._run_slash("/resume", [])
    rendered = screen_text(app, console)
    assert "No previous session found" in rendered
    assert "/resume failed: cannot read session" in rendered


def test_opening_a_view_without_messages_does_not_add_a_notification(command_app):
    app, console, _ = command_app

    def handle(_command, _rest, config, client, *_args):
        with Live(
            Text("Just a view"), console=console, screen=True,
            auto_refresh=False, redirect_stdout=False, redirect_stderr=False,
        ) as view:
            view.refresh()
        console.print()
        return config, client

    app.handle_slash = handle
    before = list(app.entries)
    app._run_slash("/help", [])
    assert app.entries == before
    assert app._idle_animation_active()


@pytest.mark.parametrize("has_conversation", [False, True])
def test_command_feedback_replaces_previous_notice(command_app, has_conversation):
    app, console, _ = command_app
    if has_conversation:
        app.add_user_message("Existing conversation")
    before = list(app.entries)
    app._run_slash("/resume", [])
    assert "No previous session found" in screen_text(app, console)

    for _ in range(3):
        app._run_slash("/setup", ["invalid"])
    rendered = screen_text(app, console)
    assert rendered.count("Unknown setup step 'invalid'.") == 1
    assert "No previous session found" not in rendered
    assert "Use /sessions" not in rendered
    assert app.entries == before
    if has_conversation:
        assert "Existing conversation" in rendered
    else:
        assert "█" in rendered


@pytest.mark.parametrize(("width", "height", "logo"), [
    (80, 24, "█"), (40, 14, "J A R V"),
])
@pytest.mark.parametrize("border", [True, False])
def test_notice_keeps_logo_and_animation_while_typing_and_resizing(
    command_app, width, height, logo, border,
):
    app, console, _ = command_app
    console.width, console.height = width, height
    app.config["headsup_border"] = border
    app._idle_anim_started_at = time.perf_counter() - 5
    started = app._idle_anim_started_at
    app.add_notice(Text("Notice one\nNotice two"))
    app.on_key(TextInput("draft"), 1)
    for current_width in (width, width + 5, width):
        console.width = current_width
        rendered = screen_text(app, console)
        assert logo in rendered
        assert "Notice one" in rendered
        assert "Notice two" in rendered
        assert "draft" in rendered
    assert app._idle_anim_started_at == started
    assert app._idle_animation_active()


def test_long_notice_scrolls_below_logo(command_app):
    app, console, _ = command_app
    app.add_notice(Text("\n".join(f"Feedback line {n:02}" for n in range(40))))
    rendered = screen_text(app, console)
    assert "█" in rendered
    assert "Feedback line 39" in rendered
    app.on_key("PAGEUP", 100)
    rendered = screen_text(app, console)
    assert "█" in rendered
    assert "Feedback line 00" in rendered
    assert "Feedback line 39" not in rendered
    app.on_key("PAGEDOWN", 100)
    assert "Feedback line 39" in screen_text(app, console)


@pytest.mark.parametrize(("width", "height"), [(80, 24), (40, 14), (80, 17)])
@pytest.mark.parametrize("border", [True, False])
def test_notices_do_not_move_or_restart_welcome_animation(
    command_app, monkeypatch, width, height, border,
):
    app, console, _ = command_app
    console.width, console.height = width, height
    app.config["headsup_border"] = border
    monkeypatch.setattr(headsup.time, "perf_counter", lambda: 10.0)
    app._idle_anim_started_at = 5.0
    frames = []
    render_intro = headsup.render_intro

    def record_intro(width, height, elapsed, **kwargs):
        frame = render_intro(width, height, elapsed, **kwargs)
        frames.append((width, height, elapsed, frame))
        return frame

    monkeypatch.setattr(headsup, "render_intro", record_intro)
    baseline = screen_text(app, console).splitlines()
    hint_row = next(i for i, line in enumerate(baseline) if "type " in line)

    def assert_stable():
        rendered = screen_text(app, console).splitlines()
        # Compare the actual screen through the logo, wave and hint, including
        # every surrounding star. Only the feedback area below may change.
        assert rendered[:hint_row + 1] == baseline[:hint_row + 1]
        assert frames[-1] == frames[0]
        assert app._idle_anim_started_at == 5.0
        return "\n".join(rendered)

    app._run_slash("/resume", [])
    assert_stable()
    app.add_notice(Text("Short notice"))
    assert "Short notice" in assert_stable()
    app.add_notice(Text("\n".join(f"Notice {n:02}" for n in range(40))))
    assert "Notice 39" in assert_stable()
    app.on_key("PAGEUP", 100)
    assert "Notice 00" in assert_stable()
    app._sync_transcript_from_history()
    assert screen_text(app, console).splitlines() == baseline


def test_replacing_notice_preserves_live_response_and_tool_slots(command_app):
    app, console, _ = command_app
    app.add_user_message("A question")
    response = app.upsert_assistant_message(None, "Initial response")
    app.upsert_live_tool("tool", Text("Initial tool"))
    app.add_notice(Text("Old notice"))
    app.upsert_assistant_message(response, "Completed response")
    app.add_notice(Text("Latest notice"))
    app.replace_live_tool("tool", Text("Completed tool"))
    rendered = screen_text(app, console)
    assert "Completed response" in rendered
    assert "Completed tool" in rendered
    assert "Latest notice" in rendered
    assert "Old notice" not in rendered
    app.add_user_message("Next question")
    rendered = screen_text(app, console)
    assert "Latest notice" not in rendered
    assert "Completed response" in rendered


def test_feedback_respects_disabled_intro(command_app):
    app, console, _ = command_app
    app.config.update(headsup_intro_logo=False, headsup_intro_stars=False)
    app.add_notice(Text("Feedback without animation"))
    rendered = screen_text(app, console)
    assert "Feedback without animation" in rendered
    assert "█" not in rendered
    assert not app._idle_animation_active()


def test_alias_confirmation_keeps_question_and_choices_together(command_app, monkeypatch):
    app, console, _ = command_app

    def answer(_label):
        rendered = screen_text(app, console)
        assert "Did you mean /new or a message?" in rendered
        assert "1 run command   2 send message: new" in rendered
        return "1"

    monkeypatch.setattr(app, "read_answer", answer)
    assert app._confirm_command_alias("/new", [], "new")


def test_final_error_keeps_failure_details(command_app):
    app, console, _ = command_app

    def run_agent(*_args, ui, **_kwargs):
        ui.show_error("Connection failed")
        return SimpleNamespace(error="Connection failed", cancelled=False)

    app.agent_import["module"] = SimpleNamespace(run_agent=run_agent)
    app._run_agent_query_now("hello")
    assert "Turn failed. Connection failed" in screen_text(app, console)
