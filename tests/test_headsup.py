import io
import re
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from conftest import make_console, wait_for
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from jarv.agent import thought_complete_indicator, tool_complete_indicator
from jarv.agent_ui import InteractiveCommandCard, RunningCommandCard
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.shell import InteractiveCommandSnapshot
from jarv.command_input import TextInput
from jarv.headsup import HeadsupAgentUI, HeadsupApp
from jarv.text_editor import initialize_text_editor


class HeadsupTests(unittest.TestCase):
    def _app(self, *, width=50, args=None, config=None):
        ready = threading.Event()
        ready.set()
        test_console, output = make_console(width=width)
        app = HeadsupApp(
            config or {"provider": "openai", "model": "test-model"},
            client=object(),
            args=args,
            agent_loader=({"module": SimpleNamespace()}, ready),
            handle_slash=lambda command, rest, config, client, args, hint: (config, client),
            maybe_command=lambda _first, _rest: None,
            render_console=test_console,
        )
        return app, test_console, output

    class _FakeLive:
        def refresh(self):
            pass

        def start(self, refresh=False):
            pass

        def stop(self):
            pass

    def _wait_for(self, predicate, timeout=1.0):
        return wait_for(predicate, timeout)

    def _entry_text(self, app: HeadsupApp) -> str:
        return "\n".join(line.plain for line in app._transcript_lines(100))

    def _rendered_text(self, app: HeadsupApp, console: Console, output: io.StringIO, *, width=80, height=12) -> str:
        output.seek(0)
        output.truncate(0)
        with patch("jarv.headsup.terminal_size", return_value=(width, height)):
            console.print(app.render())
        return output.getvalue()

    def test_render_keeps_prompt_and_footer_in_narrow_terminal(self):
        app, test_console, output = self._app(width=42)
        app.add_user_message("hello from a narrow terminal")
        app.upsert_assistant_message(None, "reply body")

        with patch("jarv.headsup.terminal_size", return_value=(42, 12)):
            test_console.print(app.render())

        rendered = output.getvalue()
        self.assertIn("jarv", rendered)
        self.assertNotIn("jarv>", rendered)
        self.assertIn("\u256d", rendered)
        self.assertIn("\u256f", rendered)
        self.assertIn("\u203a ", rendered)
        self.assertIn("hello from a narrow terminal", rendered)
        self.assertIn("Enter send", rendered)
        self.assertIn("reply body", rendered)

    def test_render_prompt_box_uses_even_panel_padding(self):
        app, test_console, output = self._app(width=150)

        rendered = self._rendered_text(app, test_console, output, width=150, height=24)
        lines = rendered.splitlines()
        top_idx = next(idx for idx, line in enumerate(lines) if line.startswith("\u2502 \u256d"))

        self.assertTrue(lines[top_idx].startswith("\u2502 \u256d"))
        self.assertTrue(lines[top_idx].endswith("\u256e \u2502"))
        self.assertTrue(lines[top_idx + 1].startswith("\u2502 \u2502"))
        self.assertTrue(lines[top_idx + 1].endswith("\u2502 \u2502"))
        self.assertTrue(lines[top_idx + 2].startswith("\u2502 \u2570"))
        self.assertTrue(lines[top_idx + 2].endswith("\u256f \u2502"))

    def test_render_fills_terminal_width_flush(self):
        app, test_console, output = self._app(width=80)

        rendered = self._rendered_text(app, test_console, output, width=80, height=24)

        widths = [cell_len(line) for line in rendered.splitlines()]
        # The border spans the full terminal width with no reserved right gap.
        self.assertLessEqual(max(widths), 80)
        self.assertEqual(max(widths), 80)

    def test_render_long_prompt_stays_within_terminal_width(self):
        app, test_console, output = self._app(width=80)
        initialize_text_editor(
            app.editor,
            "this draft is long enough to wrap and previously left stale right borders in WSL",
        )

        rendered = self._rendered_text(app, test_console, output, width=80, height=24)

        self.assertLessEqual(max(cell_len(line) for line in rendered.splitlines()), 80)
        self.assertIn("\u256d", rendered)
        self.assertIn("\u256f", rendered)

    def test_render_erases_stale_right_edge_in_terminal_frames(self):
        ready = threading.Event()
        ready.set()
        test_console, output = make_console(width=80, force_terminal=True)
        app = HeadsupApp(
            {"provider": "openai", "model": "test-model"},
            client=object(),
            args=None,
            agent_loader=({"module": SimpleNamespace()}, ready),
            handle_slash=lambda command, rest, config, client, args, hint: (config, client),
            maybe_command=lambda _first, _rest: None,
            render_console=test_console,
        )

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            test_console.print(app.render())

        self.assertIn("\x1b[0K", output.getvalue())

    def test_prompt_box_styles_typed_text_white(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "typed")

        lines = app._prompt_lines(40, max_lines=3)
        spans = [(span.start, span.end, str(span.style)) for span in lines[1].spans]
        self.assertFalse(str(lines[1].style))
        self.assertIn((0, 1, "dim cyan"), spans)
        self.assertIn((2, 7, "white"), spans)
        self.assertIn((39, 40, "dim cyan"), spans)

    def test_prompt_box_lights_up_valid_command_aqua(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "/help")

        lines = app._prompt_lines(40, max_lines=3)
        spans = [(span.start, span.end, str(span.style)) for span in lines[1].spans]
        # The "/help" token (offsets 2-7 inside the box) is repainted cyan.
        self.assertIn((2, 7, "cyan"), spans)

    def test_prompt_box_highlights_only_command_token_not_args(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "/set key value")

        lines = app._prompt_lines(40, max_lines=3)
        spans = [(span.start, span.end, str(span.style)) for span in lines[1].spans]
        # Only "/set" (offsets 2-6) goes aqua; the arguments stay white.
        self.assertIn((2, 6, "cyan"), spans)
        self.assertFalse(any(st == "cyan" and end > 6 for _s, end, st in spans))

    def test_prompt_box_leaves_unknown_command_white(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "/nope")

        lines = app._prompt_lines(40, max_lines=3)
        spans = [(span.start, span.end, str(span.style)) for span in lines[1].spans]
        self.assertIn((2, 7, "white"), spans)
        self.assertFalse(any(str(st) == "cyan" for _s, _e, st in spans))

    def test_prompt_box_paints_active_selection(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "hello world")
        app.editor["cursor"] = 0

        # Ctrl+Shift+Right selects the first word ("hello").
        app.on_key("CTRL_SHIFT_RIGHT", 1)
        self.assertEqual(app._editor_selection_span(), (0, 5))

        lines = app._prompt_lines(40, max_lines=3)
        spans = [(span.start, span.end, str(span.style)) for span in lines[1].spans]
        # The box content starts at offset 2 ("│ "), so "hello" sits at 2..7.
        self.assertTrue(
            any(st == "black on cyan" and start >= 2 for start, _e, st in spans)
        )

    def test_selection_is_replaced_by_typing_through_on_key(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "hello world")
        app.editor["cursor"] = 0
        app.on_key("CTRL_SHIFT_RIGHT", 1)  # select "hello"

        app.on_key(TextInput("hi"), 1)

        self.assertEqual(app.editor["buffer"], "hi world")
        self.assertIsNone(app.editor["selection_anchor"])

    def test_backspace_deletes_selection_not_adjacent_chip(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "")
        # A multi-line paste collapses to a chip; then select a trailing word and
        # confirm Backspace removes the selection rather than the whole chip.
        app._apply_editor_key(TextInput("one\ntwo"), 1)
        app._apply_editor_key(TextInput(" tail"), 1)
        app.on_key("SHIFT_LEFT", 1)
        app.on_key("SHIFT_LEFT", 1)  # select "il"

        app.on_key("BACKSPACE", 1)

        self.assertTrue(app.editor["buffer"].endswith(" ta"))
        self.assertIsNone(app.editor["selection_anchor"])

    def test_ctrl_c_copies_selection_instead_of_exiting(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "hello world")
        app.editor["cursor"] = 0
        app.on_key("CTRL_SHIFT_RIGHT", 1)  # select "hello"
        stopped = []
        app.stop = lambda *a, **k: stopped.append(True)

        with patch("jarv.headsup.copy_to_clipboard", return_value=True) as copy:
            app.on_interrupt()

        copy.assert_called_once_with("hello")
        # Buffer is untouched; only the selection is cleared.
        self.assertEqual(app.editor["buffer"], "hello world")
        self.assertIsNone(app.editor["selection_anchor"])
        self.assertFalse(app._exit_armed)
        self.assertFalse(stopped)
        self.assertEqual(app._prompt_notice.plain, "Copied 5 characters to clipboard.")

    def test_ctrl_c_without_selection_falls_through_to_dismiss(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "draft")

        with patch("jarv.headsup.copy_to_clipboard") as copy:
            app.on_interrupt()

        copy.assert_not_called()
        # Existing dismiss behavior: the draft is cleared on the first press.
        self.assertEqual(app.editor["buffer"], "")
        self.assertEqual(app._prompt_notice.plain, "Draft cleared.")

    def test_ctrl_c_after_copy_clears_selection_so_dismiss_resumes(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "hello world")
        app.editor["cursor"] = 0
        app.on_key("CTRL_SHIFT_RIGHT", 1)  # select "hello"

        with patch("jarv.headsup.copy_to_clipboard", return_value=True) as copy:
            app.on_interrupt()  # copies, clears selection
            app.on_interrupt()  # no selection now -> dismiss clears the draft

        copy.assert_called_once()
        self.assertEqual(app.editor["buffer"], "")
        self.assertEqual(app._prompt_notice.plain, "Draft cleared.")

    def test_ctrl_c_copies_expanded_paste_marker(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "")
        app._apply_editor_key(TextInput("one\ntwo"), 1)  # collapses to a chip
        # Select the whole chip marker (it contains spaces, so word-select alone
        # would only grab part of it).
        app.editor["selection_anchor"] = 0
        app.editor["cursor"] = len(app.editor["buffer"])

        with patch("jarv.headsup.copy_to_clipboard", return_value=True) as copy:
            app.on_interrupt()

        copy.assert_called_once_with("one\ntwo")

    def test_ctrl_c_reports_clipboard_failure(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "hello world")
        app.editor["cursor"] = 0
        app.on_key("CTRL_SHIFT_RIGHT", 1)  # select "hello"
        stopped = []
        app.stop = lambda *a, **k: stopped.append(True)

        with patch("jarv.headsup.copy_to_clipboard", return_value=False):
            app.on_interrupt()

        self.assertIsNone(app.editor["selection_anchor"])
        self.assertFalse(stopped)
        self.assertEqual(app._prompt_notice.plain, "Couldn't reach the clipboard.")

    def test_footer_hints_copy_when_selection_active(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "hello world")
        app.editor["cursor"] = 0
        app.on_key("CTRL_SHIFT_RIGHT", 1)  # select "hello"

        self.assertIn("Ctrl+C copy selection", app._footer_line(120).plain)

    def test_intro_animation_shows_on_fresh_session_and_clears_after_first_message(self):
        app, test_console, output = self._app(width=80)

        rendered = self._rendered_text(app, test_console, output, width=80, height=24)
        self.assertIn("type a message to begin", rendered)
        self.assertIn("\u2588", rendered)
        self.assertNotIn("Heads-up mode. Type /help for commands.", rendered)

        # With a live foreground display the intro hands off via a brief,
        # non-blocking dissolve outro before the transcript takes over.
        app._foreground_input_active = True
        try:
            app.add_user_message("first message")
            self.assertTrue(app._idle_anim_stop.is_set())
            self.assertGreater(app._outro_started_at, 0.0)
        finally:
            app._foreground_input_active = False

        # Once the outro finishes, the transcript fully takes over.
        app._outro_started_at = 0.0
        rendered_after = self._rendered_text(app, test_console, output, width=80, height=24)
        self.assertNotIn("type a message to begin", rendered_after)
        self.assertIn("first message", rendered_after)
        self.assertIn("Heads-up mode. Type /help for commands.", rendered_after)

    def test_live_construction_does_not_capture_completed_intro_frame(self):
        app, test_console, output = self._app(width=80)
        captured = []

        class EagerRenderLive:
            def __init__(self, *args, **kwargs):
                captured.append(kwargs["get_renderable"]())

        with (
            patch("jarv.headsup.Live", EagerRenderLive),
            patch("jarv.headsup.terminal_size", return_value=(80, 24)),
        ):
            app._build_live(app.render, test_console)

        self.assertGreater(app._idle_anim_started_at, 0.0)
        output.seek(0)
        output.truncate(0)
        test_console.print(captured[0])
        rendered = output.getvalue()
        self.assertNotIn("type a message to begin", rendered)
        self.assertNotIn("Heads-up mode. Type /help for commands.", rendered)

    def test_top_title_styles_jarv_like_settings_menu(self):
        app, test_console, output = self._app(width=80)
        with patch("jarv.headsup.terminal_size", return_value=(80, 12)):
            test_console.print(app.render())

        rendered = output.getvalue()
        self.assertIn("jarv \u25b8 heads-up", rendered)

    def test_render_shows_provider_model_effort_top_right_and_bottom_usage_status(self):
        app, test_console, output = self._app(
            width=80,
            config={"provider": "openai", "model": "test-model", "reasoning_effort": "medium"},
        )
        usage = {
            "totals": {
                "provider_cost_usd": 0.123,
                "cost_exact_request_count": 1,
            },
            "last_root_request": {
                "model": "test-model",
                "input_tokens": 125,
            },
        }

        with (
            patch("jarv.headsup.load_usage", return_value=usage),
            patch("jarv.headsup.known_context_window", return_value=1000),
            patch("jarv.headsup.terminal_size", return_value=(80, 12)),
        ):
            test_console.print(app.render())

        lines = output.getvalue().splitlines()
        self.assertIn("jarv \u25b8 heads-up", lines[0])
        self.assertIn("openai / test-model / medium", lines[0])
        self.assertNotIn("openai / test-model / medium", lines[-1])
        self.assertIn("$0.123", lines[-1])
        self.assertNotIn("cost $0.123", lines[-1])
        self.assertIn("12.5% full", lines[-1])

    def test_usage_status_styles_money_and_context_fill(self):
        app, _test_console, _output = self._app(width=80)
        usage = {
            "totals": {
                "estimated_cost_usd": 0.123,
                "cost_estimated_request_count": 1,
            },
            "last_root_request": {
                "model": "test-model",
                "input_tokens": 900,
            },
        }

        with (
            patch("jarv.headsup.load_usage", return_value=usage),
            patch("jarv.headsup.known_context_window", return_value=1000),
        ):
            status = app._usage_status(80)

        self.assertIn("est. $0.123", status.plain)
        self.assertNotIn("cost est.", status.plain)
        money_index = status.plain.index("$0.123")
        percent_index = status.plain.index("90.0% full")
        full_index = status.plain.index("full")
        self.assertTrue(
            any(span.start <= money_index < span.end and str(span.style) == "green" for span in status.spans)
        )
        self.assertTrue(
            any(
                span.start <= percent_index < span.end and str(span.style) == "bold bright_red"
                for span in status.spans
            )
        )
        self.assertTrue(any(span.start <= full_index < span.end and str(span.style) == "dim" for span in status.spans))

    def test_usage_status_shows_zero_context_for_new_session(self):
        app, _test_console, _output = self._app(width=80)
        usage = {
            "totals": {"request_count": 0},
            "last_root_request": None,
        }

        with patch("jarv.headsup.load_usage", return_value=usage):
            status = app._usage_status(80)

        self.assertIn("$0.00", status.plain)
        self.assertNotIn("cost $0.00", status.plain)
        self.assertIn("0% full", status.plain)
        self.assertNotIn("context unknown", status.plain)

    def test_usage_line_appends_to_transcript(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)
        before = len(app.entries)
        usage = Text("Usage: 1 in", style="dim")

        ui.show_usage_line(usage)

        self.assertEqual(len(app.entries), before + 1)
        self.assertEqual(app.entries[-1].kind, "usage")
        self.assertEqual(app.entries[-1].renderable.plain, "Usage: 1 in")

    def test_streaming_deltas_are_coalesced_between_refresh_frames(self):
        class FakeApp:
            def __init__(self):
                self.messages = []
                self.refresh_count = 0

            def add_user_message(self, _query):
                pass

            def upsert_assistant_message(self, index, text):
                self.messages.append(text)
                return 0 if index is None else index

            def invalidate_usage_status(self):
                pass

            def refresh(self):
                self.refresh_count += 1

        app = FakeApp()
        ui = HeadsupAgentUI(app)
        ui.start_turn("hello", {})

        with patch("jarv.headsup.time.perf_counter", side_effect=[0.0, 0.01, 0.02]):
            ui.append_stream_delta("a")
            ui.append_stream_delta("b")
            ui.append_stream_delta("c")

        self.assertEqual(app.messages, ["a"])
        ui.finish_assistant_message("abc")
        self.assertEqual(app.messages, ["a", "abc"])
        self.assertEqual(app.refresh_count, 0)

    def test_begin_assistant_message_appends_instead_of_overwriting(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)
        ui.start_turn("hello", {})

        # First streamed turn finalizes one entry.
        ui.append_stream_delta("alpha reply")
        ui.finish_assistant_message("alpha reply")
        # A tool card lands between turns, then a second turn streams text.
        app.add_tool(Text("middle tool card"))
        ui.begin_assistant_message()
        ui.append_stream_delta("omega reply")
        ui.finish_assistant_message("omega reply")

        rendered = self._entry_text(app)
        # The second message appends after the tool card rather than upserting
        # onto (and corrupting) the first turn's entry above it.
        self.assertIn("alpha reply", rendered)
        self.assertIn("omega reply", rendered)
        self.assertEqual(rendered.count("alpha reply"), 1)
        self.assertLess(rendered.index("alpha reply"), rendered.index("middle tool card"))
        self.assertLess(rendered.index("middle tool card"), rendered.index("omega reply"))

    def test_transcript_rendering_reuses_cached_entry_lines(self):
        app, _test_console, _output = self._app()
        app.add_user_message("hello")
        app.upsert_assistant_message(None, "world")

        calls = []

        def render_lines(renderable, width):
            calls.append((renderable, width))
            return [Text(getattr(renderable, "plain", str(renderable)))]

        with patch("jarv.headsup.rendered_text_lines", side_effect=render_lines):
            first = app._transcript_lines(80)
            second = app._transcript_lines(80)

        self.assertEqual([line.plain for line in first], [line.plain for line in second])
        self.assertEqual(len(calls), len(app.entries))

    def test_initial_sync_loads_saved_history_and_tool_cards(self):
        context = SimpleNamespace(
            session_id="current-session",
            history_file=Path("history-current.json"),
        )
        saved_history = [
            {"role": "user", "content": "inspect the repo"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "run_command",
                "arguments": '{"command":"rg headsup"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "jarv/headsup.py:1",
            },
            {"role": "assistant", "content": "Found it."},
        ]

        with (
            patch("jarv.headsup.prepare_session_context", return_value=context),
            patch("jarv.headsup.load_history", return_value=saved_history),
        ):
            app, _test_console, _output = self._app()
            app._sync_initial_transcript_from_history()

        rendered = self._entry_text(app)
        self.assertIn("inspect the repo", rendered)
        self.assertIn("Command", rendered)
        self.assertIn("rg headsup", rendered)
        self.assertIn("jarv/headsup.py:1", rendered)
        self.assertIn("Found it.", rendered)

    def test_initial_sync_skips_history_in_incognito(self):
        context = SimpleNamespace(
            session_id="current-session",
            history_file=Path("history-current.json"),
        )
        args = SimpleNamespace(incognito=True, new=False)

        with (
            patch("jarv.headsup.prepare_session_context", return_value=context),
            patch("jarv.headsup.load_history") as load_history,
        ):
            app, _test_console, _output = self._app(args=args)
            app._sync_initial_transcript_from_history()

        load_history.assert_not_called()
        self.assertIn("Heads-up mode.", self._entry_text(app))

    def test_new_session_is_consumed_at_headsup_startup(self):
        context = SimpleNamespace(
            session_id="new-session",
            history_file=Path("history-new.json"),
        )
        args = SimpleNamespace(incognito=False, new=True)

        with (
            patch("jarv.headsup.forget_current_session") as forget_current_session,
            patch("jarv.headsup.prepare_session_context", return_value=context),
            patch("jarv.headsup.load_history") as load_history,
        ):
            app, _test_console, _output = self._app(args=args)
            app._sync_initial_transcript_from_history()

        forget_current_session.assert_called_once_with()
        load_history.assert_not_called()
        self.assertEqual(app.session_context.session_id, "new-session")

    def test_agent_query_passes_incognito_flag(self):
        run_agent_calls = []

        def run_agent(query, config, client, **kwargs):
            run_agent_calls.append((query, config, client, kwargs))
            return SimpleNamespace(cancelled=False)

        args = SimpleNamespace(incognito=True, new=False)
        app, _test_console, _output = self._app(
            args=args,
        )
        app.agent_import["module"] = SimpleNamespace(run_agent=run_agent)

        app._run_agent_query("private prompt")

        self.assertEqual(len(run_agent_calls), 1)
        self.assertTrue(run_agent_calls[0][3]["heads_up"])
        self.assertTrue(run_agent_calls[0][3]["incognito"])
        self.assertNotIn("new_session", run_agent_calls[0][3])

    def test_live_agent_query_runs_in_background_and_prompt_remains_editable(self):
        started = threading.Event()
        release = threading.Event()
        run_agent_calls = []

        def run_agent(query, config, client, **kwargs):
            run_agent_calls.append((query, config, client, kwargs))
            started.set()
            release.wait(timeout=1.0)
            return SimpleNamespace(cancelled=False)

        app, _test_console, _output = self._app()
        app.live = self._FakeLive()
        app._foreground_input_active = True
        app.agent_import["module"] = SimpleNamespace(run_agent=run_agent)

        app._run_agent_query("first prompt")

        self.assertTrue(started.wait(timeout=1.0))
        self.assertEqual(len(run_agent_calls), 1)
        initialize_text_editor(app.editor, "next prompt")
        self.assertEqual(app.editor["buffer"], "next prompt")

        release.set()
        app._wait_for_agent_idle(timeout=1.0)
        self.assertFalse(app._agent_busy)

    def test_live_agent_query_queues_next_message_until_current_finishes(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        run_order = []

        def run_agent(query, _config, _client, **_kwargs):
            run_order.append(query)
            if query == "first":
                first_started.set()
                release_first.wait(timeout=1.0)
            if query == "second":
                second_started.set()
            return SimpleNamespace(cancelled=False)

        app, _test_console, _output = self._app()
        app.live = self._FakeLive()
        app._foreground_input_active = True
        app.agent_import["module"] = SimpleNamespace(run_agent=run_agent)

        app._run_agent_query("first")
        self.assertTrue(first_started.wait(timeout=1.0))
        app._run_agent_query("second")

        self.assertFalse(second_started.is_set())
        self.assertIn("Queued message #1.", self._entry_text(app))

        release_first.set()
        self.assertTrue(second_started.wait(timeout=1.0))
        app._wait_for_agent_idle(timeout=1.0)

        self.assertEqual(run_order, ["first", "second"])

    def test_foreground_input_owns_cancel_key_while_agent_runs(self):
        app, _test_console, _output = self._app()
        token = CancellationToken()
        app._foreground_input_active = True

        app.bind_cancel_token(token)

        # No background esc-listener thread is spawned for the foreground loop;
        # ESC-cancel is handled inline by on_key / read_answer instead.
        self.assertTrue(app._cancel_active_turn())
        self.assertTrue(token.cancelled)
        app.unbind_cancel_token()

    def test_foreground_read_answer_restores_existing_draft(self):
        app, _test_console, _output = self._app()
        app._foreground_input_active = True
        initialize_text_editor(app.editor, "draft prompt")
        result = {}

        def read_answer():
            result["answer"] = app.read_answer("answer> ")

        thread = threading.Thread(target=read_answer)
        thread.start()
        self.assertTrue(self._wait_for(lambda: app._answer_request is not None))

        initialize_text_editor(app.editor, "yes")
        app._complete_answer()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result["answer"], "yes")
        self.assertEqual(app.editor["buffer"], "draft prompt")

    def test_ask_user_replaces_waiting_card_with_answered_card(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)
        app._foreground_input_active = True
        result = {}

        def ask_user():
            result["answer"] = ui.ask_user("What next?", app.config)

        thread = threading.Thread(target=ask_user)
        thread.start()
        self.assertTrue(self._wait_for(lambda: app._answer_request is not None))

        initialize_text_editor(app.editor, "nothing")
        app._complete_answer()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result["answer"], "nothing")
        self.assertIsNone(app._answer_request)
        tool_entries = [entry for entry in app.entries if entry.kind == "tool"]
        self.assertEqual(len(tool_entries), 1)
        rendered = self._entry_text(app)
        self.assertEqual(rendered.count("Ask user"), 1)
        self.assertIn("What next?", rendered)
        self.assertIn("> nothing", rendered)
        self.assertNotIn("answer> nothing", rendered)

    def test_status_indicators_match_oneshot_glyphs(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)

        ui.complete_response_phase("Thought for 1.0 seconds.")
        ui.complete_tool_phase("Prepared 2 actions in 0.1 seconds.")

        self.assertEqual(
            app.entries[-2].renderable.plain,
            thought_complete_indicator("Thought for 1.0 seconds.").plain,
        )
        self.assertEqual(
            app.entries[-1].renderable.plain,
            tool_complete_indicator("Prepared 2 actions in 0.1 seconds.").plain,
        )

    def test_render_places_completed_status_in_transcript_order(self):
        app, test_console, output = self._app(width=80)
        app.add_user_message("Hello")
        app.upsert_status(None, thought_complete_indicator("Thought for 1.0 seconds."))

        rendered = self._rendered_text(app, test_console, output, width=80, height=12)

        self.assertLess(rendered.index("Hello"), rendered.index("Thought for 1.0 seconds."))
        self.assertEqual(
            rendered.count("Thought for 1.0 seconds."),
            1,
        )

    def test_later_status_phases_do_not_overwrite_completed_statuses(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)

        ui.complete_response_phase("Thought for 4.4 seconds.")
        ui.complete_tool_phase("Prepared 3 actions in 0.1 seconds.")
        app.add_tool(Text("Command card"))
        ui.complete_response_phase("Thought for 0.8 seconds.")
        ui.complete_tool_phase("Wrote question in 0.3 seconds.")

        rendered = self._entry_text(app)
        expected = [
            "Thought for 4.4 seconds.",
            "Prepared 3 actions in 0.1 seconds.",
            "Command card",
            "Thought for 0.8 seconds.",
            "Wrote question in 0.3 seconds.",
        ]
        positions = [rendered.index(text) for text in expected]
        self.assertEqual(positions, sorted(positions))
        for text in expected:
            self.assertEqual(rendered.count(text), 1)

    def test_sync_transcript_from_history_includes_status_records(self):
        app, _test_console, _output = self._app()
        history = [
            {"role": "user", "content": "Hello"},
            {
                "type": "status",
                "phase": "response",
                "content": "Started responding in 1.0 second.",
            },
            {"role": "assistant", "content": "Hi."},
        ]

        with patch("jarv.headsup.load_history", return_value=history):
            app._sync_transcript_from_history()

        rendered = self._entry_text(app)
        self.assertIn("Started responding in 1.0 second.", rendered)
        self.assertLess(rendered.index("Started responding"), rendered.index("Hi."))

    def test_active_wait_status_refresh_updates_elapsed_text(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)
        ui._response_started_at = 0.0
        ui._response_waiting = True
        ui._response_status_index = app.upsert_status(None, Text("old"))

        with patch("jarv.headsup.time.perf_counter", return_value=2.0):
            self.assertTrue(ui._refresh_wait_statuses())

        self.assertIn("2s", app.entries[ui._response_status_index].renderable.plain)

        status_index = ui._response_status_index
        ui.complete_response_phase("Thought for 2.0 seconds.")
        completed = app.entries[status_index].renderable.plain

        with patch("jarv.headsup.time.perf_counter", return_value=5.0):
            self.assertFalse(ui._refresh_wait_statuses())

        self.assertIsNone(ui._response_status_index)
        self.assertEqual(app.entries[status_index].renderable.plain, completed)

    def test_running_command_card_keeps_headsup_refresh_active(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)

        ui.show_tool_card(
            RunningCommandCard(
                "Start-Sleep -Seconds 10",
                "",
                "fullscreen",
                time.perf_counter(),
            )
        )
        # A running command card is an active animation the loop keeps repainting.
        self.assertTrue(ui.has_active_animation())
        app._dirty = False
        self.assertTrue(ui._refresh_wait_statuses())
        self.assertTrue(app._dirty)

        ui.show_tool_card(Text("done"))
        # Once the card is replaced with its result, nothing is animating.
        self.assertFalse(ui.has_active_animation())
        app._dirty = False
        self.assertFalse(ui._refresh_wait_statuses())
        self.assertFalse(app._dirty)

    def test_interactive_command_card_grows_in_one_headsup_slot(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)

        card = InteractiveCommandCard(
            "python game.py", "", "fullscreen", time.perf_counter()
        )
        card.seed_initial(
            InteractiveCommandSnapshot("python game.py", "hi\n", "", None, exited=False)
        )
        ui.show_tool_card(card)
        slot_count = len(app.entries)
        # A live (not yet exited) card animates and occupies one tool slot.
        self.assertTrue(ui.has_active_animation())

        # A later model step grows the SAME slot instead of appending a box.
        card.add_step(Text("stdin> go"), "there\n", 1.0, exited=False)
        ui.show_tool_card(card)
        self.assertEqual(len(app.entries), slot_count)
        self.assertTrue(ui.has_active_animation())

        # On exit the slot is finalized in place and nothing is animating.
        card.add_step(Text("stdin> quit"), "bye\n", 0.5, exited=True, exit_code=0)
        ui.show_tool_card(card)
        self.assertEqual(len(app.entries), slot_count)
        self.assertFalse(ui.has_active_animation())

    def test_thinking_card_footer_ticks_on_refresh_without_reupsert(self):
        # Regression: the "deciding next input… 0s" footer used to freeze because
        # the per-entry render cache replayed its first frame; the tick path must
        # invalidate the live tool so its clock-driven footer advances in place.
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)

        card = InteractiveCommandCard(
            "python game.py", "", "fullscreen", time.perf_counter()
        )
        card.seed_initial(
            InteractiveCommandSnapshot("python game.py", "hi\n", "", None, exited=False)
        )
        card.set_thinking(time.perf_counter() - 5.0)
        ui.show_tool_card(card)

        def footer_line() -> str:
            return next(
                (l.plain for l in app._transcript_lines(100) if "deciding next input" in l.plain),
                "",
            )

        first = footer_line()  # populates the per-entry render cache
        self.assertRegex(first, r"deciding next input…\s+5s")

        # Advance the model's think time, then drive one animation tick. Nothing
        # re-upserts the card, so only cache invalidation lets the footer recompute.
        card.set_thinking(time.perf_counter() - 9.0)
        self.assertTrue(ui._refresh_wait_statuses())
        self.assertRegex(footer_line(), r"deciding next input…\s+9s")

    def test_unbind_cancel_token_stops_wait_status_refreshes(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)
        ui._response_waiting = True
        ui._tool_waiting = True

        ui.unbind_cancel_token()

        self.assertFalse(ui._response_waiting)
        self.assertFalse(ui._tool_waiting)
        self.assertFalse(ui._refresh_wait_statuses())

    def test_bare_command_alias_uses_headsup_confirmation(self):
        calls = []
        app, _test_console, _output = self._app()
        app.maybe_command = lambda _first, _rest: (_ for _ in ()).throw(AssertionError("legacy prompt used"))

        def handle_slash(command, rest, config, client, args, hint):
            calls.append((command, rest, hint))
            return config, client

        app.handle_slash = handle_slash
        keys = ["1", "ENTER"]

        with patch("jarv.headsup._read_key_with_repeats", side_effect=lambda **kwargs: (keys.pop(0), 1)):
            app._handle_query("set model test-model")

        self.assertEqual(calls, [("/set", ["model", "test-model"], True)])

    def test_bare_command_alias_reads_choice_on_foreground_input_thread(self):
        calls = []
        app, _test_console, _output = self._app()
        app._foreground_input_active = True
        app._foreground_input_thread = threading.current_thread()
        app.maybe_command = lambda _first, _rest: (_ for _ in ()).throw(AssertionError("legacy prompt used"))

        def handle_slash(command, rest, config, client, args, hint):
            calls.append((command, rest, hint))
            return config, client

        app.handle_slash = handle_slash
        keys = ["1", "ENTER"]

        with patch("jarv.headsup._read_key_with_repeats", side_effect=lambda **kwargs: (keys.pop(0), 1)):
            app._handle_query("config")

        self.assertEqual(calls, [("/config", [], True)])
        self.assertIsNone(app._answer_request)

    def test_declined_bare_command_alias_sends_message(self):
        run_agent_calls = []
        app, _test_console, _output = self._app()
        app.maybe_command = lambda _first, _rest: (_ for _ in ()).throw(AssertionError("legacy prompt used"))

        def run_agent(query, config, client, **kwargs):
            run_agent_calls.append((query, config, client, kwargs))
            return SimpleNamespace(cancelled=False)

        app.agent_import["module"] = SimpleNamespace(run_agent=run_agent)
        keys = ["2", "ENTER"]

        with patch("jarv.headsup._read_key_with_repeats", side_effect=lambda **kwargs: (keys.pop(0), 1)):
            app._handle_query("set model test-model")

        self.assertEqual(len(run_agent_calls), 1)
        self.assertEqual(run_agent_calls[0][0], "set model test-model")

    def test_prompt_dismiss_clears_draft_then_arms_exit(self):
        app, _test_console, _output = self._app()
        app.editor["buffer"] = "draft"

        self.assertFalse(app._handle_prompt_dismiss())
        self.assertEqual(app.editor["buffer"], "")
        self.assertFalse(app._exit_armed)
        self.assertEqual(app._prompt_notice.plain, "Draft cleared.")
        self.assertFalse(app._handle_prompt_dismiss())
        self.assertTrue(app._exit_armed)
        self.assertEqual(app._prompt_notice.plain, "Press Esc or Ctrl+C again to exit.")
        self.assertTrue(app._handle_prompt_dismiss())

    def test_prompt_dismiss_clears_draft_with_transient_notice(self):
        app, _test_console, _output = self._app()
        app.editor["buffer"] = "draft"
        app._exit_armed = True

        self.assertFalse(app._handle_prompt_dismiss())

        self.assertEqual(app.editor["buffer"], "")
        self.assertTrue(app._exit_armed)
        self.assertEqual(app._prompt_notice.plain, "Draft cleared.")
        notices = [entry.renderable.plain for entry in app.entries if entry.kind == "notice"]
        self.assertNotIn("Draft cleared.", notices)

    def test_exit_prompt_notice_is_not_transcript_history(self):
        app, _test_console, _output = self._app()

        self.assertFalse(app._handle_prompt_dismiss())

        self.assertEqual(app._prompt_notice.plain, "Press Esc or Ctrl+C again to exit.")
        self.assertNotIn("Press Esc or Ctrl+C again to exit.", self._entry_text(app))

    def test_prompt_notice_clears_before_next_message(self):
        class FakeLive:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def refresh(self):
                return None

        run_agent_calls = []

        def run_agent(query, config, client, **kwargs):
            run_agent_calls.append((query, config, client, kwargs))
            return SimpleNamespace(cancelled=False)

        app, _test_console, _output = self._app()
        app._initial_history_synced = True
        app.agent_import["module"] = SimpleNamespace(run_agent=run_agent)
        keys = [
            ("ESC", 1),
            (TextInput("hi"), 1),
            ("ENTER", 1),
            (TextInput("exit"), 1),
            ("ENTER", 1),
        ]

        with (
            patch("jarv.headsup.Live", FakeLive),
            patch("jarv.headsup.disable_mouse_capture") as disable_mouse_capture,
            patch("jarv.headsup._key_available", lambda: bool(keys)),
            patch("jarv.headsup._read_key_with_repeats", side_effect=lambda **kwargs: keys.pop(0)),
        ):
            app.run()

        # on_stop tears down mouse reporting once (on_start now *enables* SGR
        # wheel reporting via enable_mouse_wheel_reporting instead).
        self.assertEqual(disable_mouse_capture.call_count, 1)
        self.assertNotEqual(getattr(app._prompt_notice, "plain", None), "Press Esc or Ctrl+C again to exit.")
        self.assertNotIn("Press Esc or Ctrl+C again to exit.", self._entry_text(app))
        self.assertEqual(run_agent_calls[0][0], "hi")

    def test_prompt_history_replays_sent_prompts_and_restores_draft(self):
        app, _test_console, _output = self._app()
        app._prompt_history = []
        app._record_prompt_history("first")
        app._record_prompt_history("second")
        app.editor["buffer"] = "draft"
        app.editor["cursor"] = len("draft")

        self.assertTrue(app._navigate_prompt_history("UP", 1))
        self.assertEqual(app.editor["buffer"], "second")

        self.assertTrue(app._navigate_prompt_history("UP", 1))
        self.assertEqual(app.editor["buffer"], "first")

        self.assertTrue(app._navigate_prompt_history("DOWN", 1))
        self.assertEqual(app.editor["buffer"], "second")

        self.assertTrue(app._navigate_prompt_history("DOWN", 1))
        self.assertEqual(app.editor["buffer"], "draft")
        self.assertIsNone(app._prompt_history_index)

    def test_multiline_paste_collapses_to_placeholder_and_expands_on_submit(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            changed, user_text = app._apply_editor_key(TextInput("first\nsecond"), 1)

        self.assertTrue(changed)
        self.assertTrue(user_text)
        # The bulky paste collapses to a single-line marker in the draft.
        self.assertEqual(app.editor["buffer"], "[Pasted text #1 +2 lines]")
        self.assertFalse(app._prompt_has_multiline_draft())

        # Submitting restores the original paste before it is dispatched.
        calls: list[str] = []
        app._run_agent_query = calls.append
        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app.on_key("ENTER", 1)

        self.assertEqual(calls, ["first\nsecond"])
        self.assertEqual(app.editor["buffer"], "")

    def test_collapsed_paste_keeps_surrounding_typed_text(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app._apply_editor_key(TextInput("alpha\nbeta\ngamma"), 1)
            for char in " do it":
                app._apply_editor_key(char, 1)

        self.assertEqual(app.editor["buffer"], "[Pasted text #1 +3 lines] do it")

        calls: list[str] = []
        app._run_agent_query = calls.append
        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app.on_key("ENTER", 1)

        self.assertEqual(calls, ["alpha\nbeta\ngamma do it"])

    def test_single_line_paste_is_not_collapsed(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app._apply_editor_key(TextInput("hello world"), 1)

        self.assertEqual(app.editor["buffer"], "hello world")

    def test_paste_in_answer_modal_is_not_collapsed(self):
        app, _test_console, _output = self._app()
        app._answer_request = {"label": "> ", "answer": None}

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app._apply_editor_key(TextInput("yes\nno"), 1)

        self.assertEqual(app.editor["buffer"], "yes\nno")

    def test_backspace_removes_whole_paste_chip(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app._apply_editor_key(TextInput("alpha\nbeta\ngamma"), 1)
            self.assertEqual(app.editor["buffer"], "[Pasted text #1 +3 lines]")
            changed, user_text = app._apply_editor_key("BACKSPACE", 1)

        self.assertTrue(changed)
        self.assertFalse(user_text)
        self.assertEqual(app.editor["buffer"], "")
        self.assertEqual(app.editor["cursor"], 0)

    def test_delete_removes_whole_paste_chip(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app._apply_editor_key(TextInput("alpha\nbeta"), 1)
            app.editor["cursor"] = 0  # in front of the chip
            app._apply_editor_key("DELETE", 1)

        self.assertEqual(app.editor["buffer"], "")

    def test_backspace_chip_keeps_surrounding_text(self):
        app, _test_console, _output = self._app()
        marker = "[Pasted text #1 +2 lines]"

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app._apply_editor_key(TextInput("alpha\nbeta"), 1)
            for char in " tail":
                app._apply_editor_key(char, 1)
            self.assertEqual(app.editor["buffer"], f"{marker} tail")
            app.editor["cursor"] = len(marker)  # just after the chip
            app._apply_editor_key("BACKSPACE", 1)

        self.assertEqual(app.editor["buffer"], " tail")

    def test_duplicate_paste_unboxes_to_one_plain_copy(self):
        app, _test_console, _output = self._app()
        block = "alpha\nbeta\ngamma"

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app._apply_editor_key(TextInput(block), 1)
            self.assertEqual(app.editor["buffer"], "[Pasted text #1 +3 lines]")
            # Cursor sits just after the chip; re-pasting the same block unboxes it.
            changed, user_text = app._apply_editor_key(TextInput(block), 1)

        self.assertTrue(changed)
        self.assertTrue(user_text)
        self.assertEqual(app.editor["buffer"], block)

        calls: list[str] = []
        app._run_agent_query = calls.append
        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app.on_key("ENTER", 1)

        self.assertEqual(calls, [block])

    def test_distinct_paste_adds_a_second_chip(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app._apply_editor_key(TextInput("alpha\nbeta"), 1)
            app._apply_editor_key(TextInput("gamma\ndelta"), 1)

        self.assertEqual(
            app.editor["buffer"],
            "[Pasted text #1 +2 lines][Pasted text #2 +2 lines]",
        )

    def test_ctrl_v_attaches_clipboard_image_as_chip(self):
        app, _test_console, _output = self._app()
        image = SimpleNamespace(path=Path("C:/tmp/jarv/image-1.png"), media_type="image/png")
        capability = SimpleNamespace(supported=True, reason="")

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)), patch(
            "jarv.headsup.read_clipboard_image", return_value=image
        ), patch("jarv.headsup.get_image_output_capability", return_value=capability):
            for char in "look: ":
                app._apply_editor_key(char, 1)
            app.on_key("CTRL_V", 1)

        self.assertEqual(app.editor["buffer"], "look: [Image #1]")
        self.assertIn("Image attached", app._prompt_notice.plain)

        # Submitting expands the chip to a file reference the read tool resolves.
        calls: list[str] = []
        app._run_agent_query = calls.append
        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            app.on_key("ENTER", 1)

        self.assertEqual(calls, [f"look: [attached image: {image.path}]"])
        self.assertEqual(app.editor["buffer"], "")

    def test_ctrl_v_image_chip_deletes_atomically(self):
        app, _test_console, _output = self._app()
        image = SimpleNamespace(path=Path("/tmp/shot.png"), media_type="image/png")
        capability = SimpleNamespace(supported=True, reason="")

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)), patch(
            "jarv.headsup.read_clipboard_image", return_value=image
        ), patch("jarv.headsup.get_image_output_capability", return_value=capability):
            app.on_key("CTRL_V", 1)
            self.assertEqual(app.editor["buffer"], "[Image #1]")
            app.on_key("BACKSPACE", 1)

        self.assertEqual(app.editor["buffer"], "")

    def test_ctrl_v_warns_when_model_lacks_image_capability(self):
        app, _test_console, _output = self._app()
        image = SimpleNamespace(path=Path("/tmp/shot.png"), media_type="image/png")
        capability = SimpleNamespace(supported=False, reason="no image capability")

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)), patch(
            "jarv.headsup.read_clipboard_image", return_value=image
        ), patch("jarv.headsup.get_image_output_capability", return_value=capability):
            app.on_key("ALT_V", 1)

        self.assertEqual(app.editor["buffer"], "[Image #1]")
        self.assertIn("can't view images", app._prompt_notice.plain)

    def test_ctrl_v_falls_back_to_clipboard_text(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)), patch(
            "jarv.headsup.read_clipboard_image", return_value=None
        ), patch("jarv.headsup.read_clipboard_text", return_value="hello world"):
            app.on_key("CTRL_V", 1)

        self.assertEqual(app.editor["buffer"], "hello world")

    def test_ctrl_v_with_empty_clipboard_sets_notice(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)), patch(
            "jarv.headsup.read_clipboard_image", return_value=None
        ), patch("jarv.headsup.read_clipboard_text", return_value=None):
            app.on_key("CTRL_V", 1)

        self.assertEqual(app.editor["buffer"], "")
        self.assertIn("Nothing to paste", app._prompt_notice.plain)

    def test_ctrl_v_is_ignored_while_answering(self):
        app, _test_console, _output = self._app()
        app._answer_request = {"label": "> ", "answer": None}
        reads: list[str] = []

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)), patch(
            "jarv.headsup.read_clipboard_image", side_effect=lambda: reads.append("read")
        ):
            app.on_key("CTRL_V", 1)

        self.assertEqual(reads, [])
        self.assertEqual(app.editor["buffer"], "")

    def test_multiline_prompt_arrows_move_between_editor_rows(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "first\nsecond", multiline=True)
        app.editor["cursor"] = 0

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            changed, user_text = app._apply_editor_key("DOWN", 1)

        self.assertFalse(changed)
        self.assertFalse(user_text)
        self.assertEqual(app.editor["cursor"], len("first\n"))
        self.assertTrue(app._prompt_has_multiline_draft())

    def test_multiline_query_bypasses_command_detection(self):
        app, _test_console, _output = self._app()
        calls = []
        app._run_agent_query = calls.append

        result = app._handle_query("/help\nexplain this text")

        self.assertIsNone(result)
        self.assertEqual(calls, ["/help\nexplain this text"])

    def test_slash_menu_renders_above_input_box_for_partial_command(self):
        app, test_console, output = self._app(width=80)
        initialize_text_editor(app.editor, "/se")

        rendered = self._rendered_text(app, test_console, output, width=80, height=20)

        self.assertIn("/settings", rendered)
        self.assertIn("/setup", rendered)
        self.assertIn("Open common controls", rendered)
        # The input box itself still renders beneath the menu.
        self.assertIn("╭", rendered)

    def test_slash_menu_overlays_intro_without_pushing_it_up(self):
        # Opening the menu must not shrink the transcript/intro region: it floats
        # over the bottom of the body instead of displacing rows, so the
        # starfield and JARV logo stay put. We assert render_intro is asked for
        # the same height whether the menu is open or closed.
        app, test_console, output = self._app(width=80)
        heights: list[int] = []

        def fake_intro(width, height, *args, **kwargs):
            heights.append(height)
            return [Text("") for _ in range(height)]

        with patch("jarv.headsup.render_intro", side_effect=fake_intro):
            self._rendered_text(app, test_console, output, width=80, height=20)
            initialize_text_editor(app.editor, "/se")
            self._rendered_text(app, test_console, output, width=80, height=20)

        self.assertEqual(len(heights), 2)
        self.assertEqual(heights[0], heights[1])

    def test_slash_menu_open_does_not_shift_the_transcript(self):
        # Opening the popup used to hand the footer-hint row to the transcript,
        # dragging every line down one row. The transcript must keep the same
        # height (and its lines the same screen rows) whether the popup is open
        # or closed; the freed row carries the popup's own key hints instead.
        app, test_console, output = self._app(width=80)
        anchor = "anchor " * 10  # long enough to peek out past the popup
        app.add_user_message(anchor.strip())

        closed = self._rendered_text(app, test_console, output, width=80, height=20).splitlines()
        initialize_text_editor(app.editor, "/se")
        opened = self._rendered_text(app, test_console, output, width=80, height=20).splitlines()

        def anchor_rows(lines):
            return [idx for idx, line in enumerate(lines) if "anchor" in line]

        self.assertTrue(anchor_rows(closed))
        self.assertEqual(anchor_rows(closed), anchor_rows(opened))

    def test_slash_menu_shows_key_hints_beside_popup_when_room_allows(self):
        # On a wide terminal the row beside the popup's bottom edge teaches the
        # menu keys -- in the very cells the footer hints use while it's closed.
        app, test_console, output = self._app(width=160)

        closed = self._rendered_text(app, test_console, output, width=160, height=20)
        self.assertIn("Enter send", closed)

        initialize_text_editor(app.editor, "/se")
        opened = self._rendered_text(app, test_console, output, width=160, height=20)
        self.assertIn("Tab complete", opened)
        self.assertNotIn("Enter send", opened)

    def test_slash_menu_slash_column_aligns_with_input_draft(self):
        # The popup's commands and the input field's draft share a column: the
        # selection caret lives in the box gutter, so every "/" lines up.
        from jarv.tui_frame import compute_layout

        app, _test_console, _output = self._app(width=80)
        initialize_text_editor(app.editor, "/se")
        layout = compute_layout(80, 20)
        width = layout.inner_width

        popup = app._slash_menu_box(width, layout)
        field = app._prompt_lines(width, max_lines=layout.max_prompt_rows, menu_open=True)

        selected_row = popup[1].plain
        unselected_row = popup[2].plain
        self.assertEqual(selected_row[1], "›")   # caret in the gutter
        self.assertEqual(selected_row.index("/"), 2)
        self.assertEqual(unselected_row[1], " ")
        self.assertEqual(unselected_row.index("/"), 2)
        self.assertEqual(field[0].plain.index("/"), 2)

    def test_slash_menu_is_a_compact_box_docked_onto_the_input_field(self):
        # The popup is compact and left-aligned (no wider than the field, sized to
        # its rows, open at the bottom), and the footer docks it onto the
        # full-width field: a tee (┴) under the popup's right border, then the
        # field's own top edge. Every border shares the one dim-cyan style.
        from jarv.tui_frame import box_tab_top, compute_layout

        app, _test_console, _output = self._app(width=80)
        initialize_text_editor(app.editor, "/se")
        layout = compute_layout(80, 20)
        width = layout.inner_width

        popup = app._slash_menu_box(width, layout)
        popup_width = cell_len(popup[0].plain)
        footer = box_tab_top(width, popup_width)
        field = app._prompt_lines(width, max_lines=layout.max_prompt_rows, menu_open=True)

        # Compact + left-aligned: a uniform box, narrower than the field, opening
        # at the top with no bottom edge of its own.
        self.assertTrue(any("/settings" in line.plain for line in popup))
        self.assertLess(popup_width, width)
        self.assertTrue(all(cell_len(line.plain) == popup_width for line in popup))
        self.assertEqual(popup[0].plain[0], "╭")
        self.assertNotIn("╰", "".join(line.plain for line in popup))
        # The footer docks the popup onto the field: ├ … ┴ (under the popup's right
        # border) … ╮, spanning the full width.
        self.assertEqual(cell_len(footer.plain), width)
        self.assertEqual(footer.plain[0], "├")
        self.assertEqual(footer.plain[popup_width - 1], "┴")
        self.assertEqual(footer.plain[-1], "╮")
        # The field drops its own top edge and closes the box at full width.
        self.assertEqual(field[0].plain[0], "│")
        self.assertEqual(field[-1].plain[0], "╰")
        self.assertEqual(cell_len(field[-1].plain), width)
        # The popup top, the docking edge, and the field's bottom share one border
        # colour -- carried as a span so it survives the overlay that paints the
        # top edge onto the body (a base style is dropped there and shows white).
        def edge_style(edge):
            return " ".join(str(span.style) for span in edge.spans)

        self.assertEqual(edge_style(popup[0]), "dim cyan")
        self.assertEqual(edge_style(footer), "dim cyan")
        self.assertEqual(edge_style(field[-1]), "dim cyan")

    def test_slash_menu_summaries_are_evenly_dim_across_rows(self):
        # The highlighted row is marked by its caret and brighter command, not by a
        # white summary -- so no row's summary reads as a different colour.
        app, _test_console, _output = self._app(width=80)
        initialize_text_editor(app.editor, "/se")
        matches = app._slash_menu_matches()

        for selected in (True, False):
            row = app._slash_menu_row(
                matches[0], selected, "se", name_col=12,
                gap=2, width=60,
            )
            summary_styles = {
                str(span.style) for span in row.spans
                if "Open" in row.plain[span.start:span.end]
                or "Run" in row.plain[span.start:span.end]
            }
            self.assertNotIn("white", " ".join(summary_styles))

    def test_slash_menu_arrows_drive_highlight_not_prompt_history(self):
        app, _test_console, _output = self._app()
        app._prompt_history = ["earlier message"]
        initialize_text_editor(app.editor, "/se")

        self.assertEqual(app._slash_menu_index, 0)
        app.on_key("DOWN", 1)
        self.assertEqual(app._slash_menu_index, 1)
        app.on_key("UP", 1)
        self.assertEqual(app._slash_menu_index, 0)
        # The draft is untouched: arrows moved the menu, not prompt history.
        self.assertEqual(app.editor["buffer"], "/se")
        self.assertIsNone(app._prompt_history_index)

    def test_slash_menu_up_at_window_bottom_moves_highlight_not_window(self):
        # Regression: with more matches than fit, pressing UP while the highlight
        # sits on the bottom visible row moves the highlight up *within* the
        # window -- it must not scroll the whole window up by one.
        from jarv.headsup import _SLASH_MENU_MAX_ROWS
        from jarv.tui_frame import compute_layout

        app, _test_console, _output = self._app(width=80)
        initialize_text_editor(app.editor, "/")  # empty query -> every command
        layout = compute_layout(80, 24)
        width = layout.inner_width

        # The live loop renders after each key; mirror that so the scroll anchor
        # advances as it would on screen.
        app._slash_menu_box(width, layout)
        self.assertGreater(len(app._slash_menu_matches()), _SLASH_MENU_MAX_ROWS)

        # Page down until the window has actually scrolled, leaving the highlight
        # on the bottom visible row.
        while app._slash_menu_scroll == 0:
            app.on_key("DOWN", 1)
            app._slash_menu_box(width, layout)
        anchor = app._slash_menu_scroll
        index = app._slash_menu_index

        app.on_key("UP", 1)
        app._slash_menu_box(width, layout)

        self.assertEqual(app._slash_menu_index, index - 1)  # highlight moved up
        self.assertEqual(app._slash_menu_scroll, anchor)    # window stayed put

    def test_slash_menu_more_tail_counts_down_and_marks_bottom(self):
        # Regression: the "+N more" tail summarises entries *below* the window, so
        # it shrinks as the user scrolls down. At the bottom it must not keep
        # reporting every off-window entry, nor vanish (which would shift the box
        # height) -- it stays put and reads "no more".
        from jarv.headsup import _SLASH_MENU_MAX_ROWS
        from jarv.tui_frame import compute_layout

        app, _test_console, _output = self._app(width=80)
        initialize_text_editor(app.editor, "/")  # empty query -> every command
        layout = compute_layout(80, 24)
        width = layout.inner_width

        total = len(app._slash_menu_matches())
        self.assertGreater(total, _SLASH_MENU_MAX_ROWS)

        def more_count(popup):
            for line in popup:
                match = re.search(r"\+(\d+) more", line.plain)
                if match:
                    return int(match.group(1))
            return None

        def has_no_more(popup):
            return any("no more" in line.plain for line in popup)

        # At the top the tail reports the entries hidden below the window.
        top_popup = app._slash_menu_box(width, layout)
        top = more_count(top_popup)
        self.assertIsNotNone(top)
        self.assertGreater(top, 0)

        # Drive the highlight to the very last match, rendering after each key as
        # the live loop does so the scroll anchor advances.
        seen_smaller = False
        while app._slash_menu_index < total - 1:
            app.on_key("DOWN", 1)
            current = more_count(app._slash_menu_box(width, layout))
            if current is not None and current < top:
                seen_smaller = True

        # The tail counted down on the way, then became the "no more" marker -- the
        # row is still there, so the box keeps the height it had while scrolling.
        self.assertTrue(seen_smaller)
        bottom_popup = app._slash_menu_box(width, layout)
        self.assertIsNone(more_count(bottom_popup))
        self.assertTrue(has_no_more(bottom_popup))
        self.assertEqual(len(bottom_popup), len(top_popup))
        # The final command is still on screen alongside the marker.
        last_label = app._slash_menu_matches()[-1].display
        self.assertIn(last_label, "\n".join(line.plain for line in bottom_popup))

    def test_slash_menu_tab_completes_without_running(self):
        # Tab is completion only -- it fills the box with the highlighted
        # command but never executes it; Enter is the key that runs.
        app, _test_console, _output = self._app()
        handled = []
        app._handle_query = handled.append
        initialize_text_editor(app.editor, "/sett")

        app.on_key("TAB", 1)

        self.assertEqual(handled, [])
        self.assertEqual(app.editor["buffer"], "/settings")

    def test_slash_menu_tab_fills_box_for_parameterized_command(self):
        app, _test_console, _output = self._app()
        handled = []
        app._handle_query = handled.append
        initialize_text_editor(app.editor, "/usage")

        app.on_key("TAB", 1)

        self.assertEqual(handled, [])
        self.assertEqual(app.editor["buffer"], "/usage ")
        # The menu stays open, moving on to the command's argument choices.
        self.assertEqual(
            [entry.name for entry in app._slash_menu_matches()],
            ["session", "day", "week", "month", "all"],
        )

    def test_slash_menu_enter_accepts_highlighted_entry(self):
        app, _test_console, _output = self._app()
        handled = []
        app._handle_query = handled.append
        initialize_text_editor(app.editor, "/sett")

        app.on_key("ENTER", 1)

        self.assertEqual(handled, ["/settings"])
        self.assertEqual(app.editor["buffer"], "")

    def test_slash_menu_enter_runs_complete_parameterized_command(self):
        app, _test_console, _output = self._app()
        handled = []
        app._handle_query = handled.append
        # "/usage" accepts an optional period argument, but a fully typed command
        # with no arguments runs on Enter rather than gaining a trailing space.
        initialize_text_editor(app.editor, "/usage")

        app.on_key("ENTER", 1)

        self.assertEqual(handled, ["/usage"])
        self.assertEqual(app.editor["buffer"], "")
        self.assertEqual(app._prompt_history[-1], "/usage")

    def test_slash_menu_argument_mode_filters_choices(self):
        # A command with declared argument choices keeps the menu open on the
        # first argument token and filters the choices the same way.
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "/usage d")

        self.assertEqual(app._slash_menu_context(), ("usage", "d"))
        self.assertEqual(
            [entry.name for entry in app._slash_menu_matches()], ["day"]
        )

    def test_slash_menu_argument_mode_enter_runs_completed_command(self):
        app, _test_console, _output = self._app()
        handled = []
        app._handle_query = handled.append
        initialize_text_editor(app.editor, "/usage d")

        app.on_key("ENTER", 1)

        self.assertEqual(handled, ["/usage day"])
        self.assertEqual(app.editor["buffer"], "")
        self.assertEqual(app._prompt_history[-1], "/usage day")

    def test_slash_menu_argument_mode_enter_submits_bare_command_untyped(self):
        # "/usage " with nothing typed at the argument position must not
        # force-pick a highlighted choice -- Enter submits the draft as typed.
        app, _test_console, _output = self._app()
        handled = []
        app._handle_query = handled.append
        initialize_text_editor(app.editor, "/usage ")

        self.assertTrue(app._slash_menu_matches())
        app.on_key("ENTER", 1)

        self.assertEqual(handled, ["/usage "])
        self.assertEqual(app.editor["buffer"], "")

    def test_slash_menu_argument_mode_set_key_expects_a_value(self):
        # Accepting a /set key completes "/set <key> " (value still expected)
        # rather than running; the menu then closes for the free-form value.
        app, _test_console, _output = self._app()
        handled = []
        app._handle_query = handled.append
        initialize_text_editor(app.editor, "/set mod")

        matches = app._slash_menu_matches()
        self.assertTrue(any(entry.name == "model" for entry in matches))
        while app._slash_menu_matches()[app._slash_menu_index].name != "model":
            app.on_key("DOWN", 1)
        app.on_key("ENTER", 1)

        self.assertEqual(handled, [])
        self.assertEqual(app.editor["buffer"], "/set model ")
        self.assertEqual(app._slash_menu_matches(), [])

    def test_slash_menu_inactive_for_free_form_arguments(self):
        # Commands without declared choices close the menu at the first space,
        # as does a completed argument followed by a space.
        app, _test_console, _output = self._app()

        initialize_text_editor(app.editor, "/btw how do I")
        self.assertIsNone(app._slash_menu_context())
        self.assertEqual(app._slash_menu_matches(), [])

        initialize_text_editor(app.editor, "/usage day ")
        self.assertIsNone(app._slash_menu_context())
        self.assertEqual(app._slash_menu_matches(), [])

    def test_slash_menu_esc_dismisses_until_the_draft_changes(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "/se")
        self.assertTrue(app._slash_menu_matches())

        app.on_key("ESC", 1)

        # The popup is gone but the draft survives -- Esc closed the menu, it
        # did not fall through to the draft-clearing prompt dismissal.
        self.assertEqual(app._slash_menu_matches(), [])
        self.assertEqual(app.editor["buffer"], "/se")
        self.assertFalse(app._exit_armed)

        # Typing edits the draft, which reopens the menu.
        app.on_key(TextInput("t"), 1)
        self.assertEqual(app.editor["buffer"], "/set")
        self.assertTrue(app._slash_menu_matches())

    def test_slash_menu_selected_command_is_aqua_including_slash(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "/se")
        matches = app._slash_menu_matches()
        self.assertTrue(matches)

        row = app._slash_menu_row(
            matches[0],
            selected=True,
            query="se",
            name_col=12,
            gap=2,
            width=60,
        )
        display = matches[0].display
        # The command starts the row (the caret lives in the box gutter, not the
        # row) and, leading "/" included, is painted in the cyan family — never
        # bright white.
        command_spans = [
            str(span.style)
            for span in row.spans
            if span.start < len(display)
        ]
        self.assertTrue(command_spans)
        for style in command_spans:
            self.assertIn("cyan", style)
            self.assertNotIn("white", style)

    def test_mouse_wheel_scrolls_transcript_without_prompt_history_navigation(self):
        class FakeLive:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def refresh(self):
                return None

        app, _test_console, _output = self._app()
        app._initial_history_synced = True
        app._prompt_history = ["previous prompt"]
        initialize_text_editor(app.editor, "exit")
        keys = [("MOUSE_WHEEL_UP", 2), ("ENTER", 1)]

        with (
            patch("jarv.headsup.Live", FakeLive),
            patch("jarv.headsup.enable_mouse_wheel_reporting") as enable_mouse_wheel_reporting,
            patch("jarv.headsup.disable_mouse_capture") as disable_mouse_capture,
            patch("jarv.headsup._key_available", lambda: bool(keys)),
            patch("jarv.headsup._read_key_with_repeats", side_effect=lambda **kwargs: keys.pop(0)),
        ):
            app.run()

        # on_start turns SGR mouse-wheel reporting on; on_stop tears it down. The
        # wheel scrolls the transcript (3 lines x repeat 2) and never touches
        # prompt-history navigation.
        self.assertGreaterEqual(enable_mouse_wheel_reporting.call_count, 1)
        self.assertEqual(disable_mouse_capture.call_count, 1)
        self.assertEqual(app.scroll_offset, 6)
        self.assertIsNone(app._prompt_history_index)

    def test_streamed_content_preserves_scroll_offset_when_scrolled_up(self):
        app, _test_console, _output = self._app()
        # Pretend the user has scrolled up to read earlier history.
        app.scroll_offset = 4

        # A tool card and streamed assistant deltas must not yank the view back to
        # the bottom while the user is reading earlier content.
        app.add_tool(Text("tool output"))
        self.assertEqual(app.scroll_offset, 4)
        index = app.upsert_assistant_message(None, "streamed")
        app.upsert_assistant_message(index, "streamed reply")
        self.assertEqual(app.scroll_offset, 4)

        # A new turn always jumps back to the bottom so the user follows it.
        app.add_user_message("next question")
        self.assertEqual(app.scroll_offset, 0)

    def test_sgr_mouse_text_fragment_does_not_enter_prompt(self):
        app, _test_console, _output = self._app()

        changed, user_text = app._apply_editor_key(TextInput("[<35;62;15M"), 1)

        self.assertFalse(changed)
        self.assertFalse(user_text)
        self.assertEqual(app.editor["buffer"], "")

    def test_sgr_mouse_text_fragment_is_stripped_from_batched_prompt_text(self):
        app, _test_console, _output = self._app()

        changed, user_text = app._apply_editor_key(TextInput("hi[<35;62;15Mthere"), 1)

        self.assertTrue(changed)
        self.assertTrue(user_text)
        self.assertEqual(app.editor["buffer"], "hithere")

    def test_split_sgr_mouse_text_fragment_is_removed_from_prompt(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "[")

        changed, user_text = app._apply_editor_key(TextInput("<35;62;15M"), 1)

        self.assertTrue(changed)
        self.assertFalse(user_text)
        self.assertEqual(app.editor["buffer"], "")

    def test_prompt_history_loads_user_messages_from_synced_history(self):
        app, _test_console, _output = self._app()
        updated_history = [
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"},
        ]

        with patch("jarv.headsup.load_history", return_value=updated_history):
            app._sync_transcript_from_history()

        self.assertEqual(app._prompt_history, ["first", "second"])

    def test_read_answer_esc_cancels_turn(self):
        app, _test_console, _output = self._app()
        token = CancellationToken()
        app.bind_cancel_token(token)
        keys = ["ESC"]

        with (
            patch("jarv.headsup._read_key_with_repeats", side_effect=lambda **kwargs: (keys.pop(0), 1)),
            self.assertRaises(TurnCancelled),
        ):
            app.read_answer("answer> ")

        self.assertTrue(token.cancelled)
        app.unbind_cancel_token()

    def test_read_only_slash_commands_use_fullscreen_handoff(self):
        calls = []

        class FakeLive:
            def stop(self):
                calls.append("stop")

            def refresh(self):
                calls.append("refresh")

            def start(self, refresh=False):
                calls.append(("start", refresh))

        def handle_slash(command, rest, config, client, args, hint):
            calls.append((command, rest, hint))
            return config, client

        app, _test_console, _output = self._app()
        app.handle_slash = handle_slash
        app.live = FakeLive()

        for command in ("/help", "/about", "/usage", "/config"):
            with self.subTest(command=command):
                calls.clear()
                app._run_slash(command, [])
                self.assertEqual(calls[:2], ["stop", (command, [], True)])
                self.assertEqual(calls[2], ("start", True))

    def test_fullscreen_slash_command_temporarily_stops_live(self):
        calls = []
        alt_screen_calls = []

        class FakeLive:
            def stop(self):
                calls.append("stop")
                app.console.set_alt_screen(False)

            def start(self, refresh=False):
                calls.append(("start", refresh))
                app.console.set_alt_screen(True)

            def refresh(self):
                calls.append("refresh")

        def handle_slash(command, rest, config, client, args, hint):
            calls.append((command, rest, hint))
            app.console.set_alt_screen(True)
            app.console.set_alt_screen(False)
            return config, client

        app, _test_console, _output = self._app()
        original_set_alt_screen = app.console.set_alt_screen

        def record_alt_screen(enable=True):
            alt_screen_calls.append(enable)
            return original_set_alt_screen(enable)

        app.console.set_alt_screen = record_alt_screen
        app.handle_slash = handle_slash
        app.live = FakeLive()

        app._run_slash("/history", [])

        self.assertEqual(calls[:2], ["stop", ("/history", [], True)])
        self.assertEqual(calls[2], ("start", True))
        # The alt screen is held steady across the nested view: none of the
        # Live stop/start or nested toggles re-enter it, so no redundant
        # \x1b[?1049h/l control codes reach the terminal (a burst of those while
        # the window is resized desyncs ConPTY's size tracking on Windows).
        self.assertEqual(alt_screen_calls, [])

    def test_all_supported_slash_commands_route_cleanly_in_headsup(self):
        cases = [
            ("/setup", ["provider"], True),
            ("/help", [], True),
            ("/about", [], True),
            ("/new", [], False),
            ("/archive", [], False),
            ("/session", ["abc123"], False),
            ("/session", [], True),
            ("/sessions", ["abc123"], False),
            ("/sessions", [], True),
            ("/history", [], True),
            ("/usage", ["day"], True),
            ("/set", ["model", "test-model"], False),
            ("/unset", ["model"], False),
            ("/config", [], True),
            ("/settings", [], True),
            ("/undo", ["2"], False),
            ("/redo", ["2"], False),
        ]

        for command, rest, fullscreen in cases:
            with self.subTest(command=command, rest=rest):
                calls = []

                class FakeLive:
                    def stop(self):
                        calls.append("stop")

                    def start(self, refresh=False):
                        calls.append(("start", refresh))

                    def refresh(self):
                        calls.append("refresh")

                config = {"provider": "openai", "model": "before"}
                refreshed = {"provider": "openai", "model": command}

                def handle_slash(command_arg, rest_arg, current_config, current_client, args, hint):
                    calls.append((command_arg, rest_arg, current_config, current_client, args, hint))
                    app.console.print(f"handled {command_arg}")
                    return refreshed, "client-after"

                app, _test_console, _output = self._app()
                app.config = config
                app.client = "client-before"
                app.args = object()
                app.handle_slash = handle_slash
                app.live = FakeLive()

                app._run_slash(command, rest)

                self.assertEqual(app.config, refreshed)
                self.assertEqual(app.client, "client-after")
                self.assertIn(
                    (command, rest, config, "client-before", app.args, True),
                    calls,
                )
                if fullscreen:
                    self.assertEqual(calls[0], "stop")
                    self.assertEqual(calls[-1], ("start", True))
                    self.assertFalse(
                        any(
                            getattr(entry.renderable, "plain", "") == f"handled {command}"
                            for entry in app.entries
                        )
                    )
                else:
                    self.assertNotIn("stop", calls)
                    self.assertNotIn(("start", True), calls)
                    if command == "/new":
                        self.assertFalse(
                            any(
                                getattr(entry.renderable, "plain", "").strip()
                                == f"handled {command}"
                                for entry in app.entries
                            )
                        )
                    else:
                        self.assertTrue(
                            any(
                                getattr(entry.renderable, "plain", "").strip()
                                == f"handled {command}"
                                for entry in app.entries
                            )
                        )

    def test_update_runs_on_worker_thread_with_live_status(self):
        from jarv.commands import UpdateOutcome

        handled = []
        app, _test_console, _output = self._app()
        app.handle_slash = lambda *call: handled.append(call)

        def fake_perform_update(stage):
            stage("Installing v9.9.9 into active Python environment")
            return UpdateOutcome("updated", "Updated successfully.", latest="9.9.9")

        with patch("jarv.commands.perform_update", fake_perform_update):
            app._run_slash("/update", [])
            self.assertTrue(self._wait_for(lambda: app._update_task is None))

        # /update never goes through the blocking captured-output slash path.
        self.assertEqual(handled, [])
        # The whole run lives in one status entry that resolves in place.
        self.assertEqual(
            sum(1 for entry in app.entries if entry.kind == "status"), 1
        )
        text = self._entry_text(app)
        self.assertIn("Updated to v9.9.9", text)
        self.assertIn("restart jarv", text)

    def test_update_animates_spinner_while_stage_runs(self):
        from jarv.commands import UpdateOutcome

        app, _test_console, _output = self._app()
        release = threading.Event()

        def slow_update(stage):
            release.wait(timeout=5.0)
            return UpdateOutcome("current", "Already up to date.", detail="v1.0.0")

        with patch("jarv.commands.perform_update", slow_update):
            app._run_slash("/update", [])
            try:
                first = self._entry_text(app)
                self.assertIn("Checking for updates", first)
                app._last_wait_tick = 0.0
                with patch(
                    "jarv.headsup.time.perf_counter",
                    return_value=time.perf_counter() + 5.0,
                ):
                    app.on_tick()
                # The loop tick advanced the elapsed timer past zero.
                self.assertRegex(
                    self._entry_text(app), r"Checking for updates…\s+[1-9]\d*s"
                )
            finally:
                release.set()
            self.assertTrue(self._wait_for(lambda: app._update_task is None))
        self.assertIn("Already up to date. (v1.0.0)", self._entry_text(app))

    def test_update_failure_reports_error_and_clears_task(self):
        app, _test_console, _output = self._app()

        def broken_update(_stage):
            raise RuntimeError("network exploded")

        with patch("jarv.commands.perform_update", broken_update):
            app._run_slash("/update", [])
            self.assertTrue(self._wait_for(lambda: app._update_task is None))

        text = self._entry_text(app)
        self.assertIn("Update failed.", text)
        self.assertIn("network exploded", text)

    def test_second_update_while_running_is_rejected(self):
        from jarv.commands import UpdateOutcome

        app, _test_console, _output = self._app()
        release = threading.Event()

        def slow_update(_stage):
            release.wait(timeout=5.0)
            return UpdateOutcome("updated", "Updated successfully.", latest="9.9.9")

        with patch("jarv.commands.perform_update", slow_update):
            app._run_slash("/update", [])
            try:
                self.assertIsNotNone(app._update_task)
                app._run_slash("/update", [])
                self.assertIn(
                    "An update is already running.", self._entry_text(app)
                )
            finally:
                release.set()
            self.assertTrue(self._wait_for(lambda: app._update_task is None))

    def test_quick_slash_command_does_not_capture_live_frame(self):
        calls = []

        class FakeLive:
            def stop(self):
                calls.append("stop")
                app.console.print("live frame should not be captured")

            def start(self, refresh=False):
                calls.append(("start", refresh))

            def refresh(self):
                calls.append("refresh")

        def handle_slash(command, rest, config, client, args, hint):
            app.console.print("command output")
            return config, client

        app, _test_console, _output = self._app()
        app.handle_slash = handle_slash
        app.live = FakeLive()

        app._run_slash("/set", ["model", "test-model"])

        rendered = self._entry_text(app)
        self.assertNotIn("stop", calls)
        self.assertNotIn(("start", True), calls)
        self.assertIn("command output", rendered)
        self.assertNotIn("live frame should not be captured", rendered)

    def test_quick_slash_command_suspends_live_render_hook_only(self):
        calls = []

        class FakeLive:
            def refresh(self):
                calls.append("refresh")

        def handle_slash(command, rest, config, client, args, hint):
            app.console.print("command output")
            return config, client

        app, _test_console, _output = self._app()
        app.handle_slash = handle_slash
        app.live = FakeLive()
        app.console.push_render_hook(app.live)

        app._run_slash("/set", ["model", "test-model"])

        self.assertEqual(app.console._render_hooks[-1], app.live)
        self.assertIn("command output", self._entry_text(app))
        self.assertIn("refresh", calls)

    def test_new_refreshes_session_context_and_clears_visible_transcript(self):
        app, test_console, output = self._app(width=80)
        old_context = SimpleNamespace(
            session_id="old-session",
            history_file=Path("history-old.json"),
        )
        new_context = SimpleNamespace(
            session_id="new-session",
            history_file=Path("history-new.json"),
        )
        app.session_context = old_context
        app.usage_path = Path("usage-old.json")
        app.add_user_message("old visible message")
        app.upsert_assistant_message(None, "old visible reply")

        def handle_slash(command, rest, config, client, args, hint):
            app.console.print("new session ready")
            return config, client

        app.handle_slash = handle_slash

        with (
            patch("jarv.headsup.prepare_session_context", return_value=new_context),
            patch("jarv.headsup.load_history", return_value=[]),
        ):
            app._run_slash("/new", [])

        rendered = self._entry_text(app)
        screen = self._rendered_text(app, test_console, output, height=24)
        self.assertEqual(app.session_context.session_id, "new-session")
        self.assertEqual(app.usage_path, Path("usage-new.json"))
        self.assertNotIn("new session ready", rendered)
        self.assertNotIn("new session ready", screen)
        self.assertNotIn("Heads-up mode. Type /help for commands.", screen)
        self.assertIn("\u2588", screen)
        self.assertNotIn("old visible message", rendered)
        self.assertNotIn("old visible reply", rendered)
        self.assertFalse(app._idle_anim_stop.is_set())

    def test_new_restarts_idle_animation_in_active_headsup(self):
        app, _test_console, _output = self._app()
        old_context = SimpleNamespace(
            session_id="old-session",
            history_file=Path("history-old.json"),
        )
        new_context = SimpleNamespace(
            session_id="new-session",
            history_file=Path("history-new.json"),
        )
        app.session_context = old_context
        app.usage_path = Path("usage-old.json")
        app.live = self._FakeLive()
        app._foreground_input_active = True
        app.add_user_message("old visible message")
        self.assertTrue(app._idle_anim_stop.is_set())

        def handle_slash(command, rest, config, client, args, hint):
            app.console.print("new session ready")
            return config, client

        app.handle_slash = handle_slash

        with (
            patch("jarv.headsup.prepare_session_context", return_value=new_context),
            patch("jarv.headsup.load_history", return_value=[]),
        ):
            try:
                app._run_slash("/new", [])
                # No background thread now: the loop's on_tick drives the intro.
                # /new should re-arm the animation (stop cleared, timer reset).
                self.assertFalse(app._idle_anim_stop.is_set())
                self.assertGreater(app._idle_anim_started_at, 0.0)
                self.assertTrue(app._idle_animation_active())
            finally:
                app._idle_anim_stop.set()
                app._foreground_input_active = False

    def test_new_on_already_empty_session_still_shows_intro(self):
        app, test_console, output = self._app(width=80)
        context = SimpleNamespace(
            session_id="current-session",
            history_file=Path("history-current.json"),
        )
        app.session_context = context
        app.usage_path = Path("usage-current.json")

        def handle_slash(command, rest, config, client, args, hint):
            app.console.print("Already on a new session.")
            return config, client

        app.handle_slash = handle_slash

        with (
            patch("jarv.headsup.prepare_session_context", return_value=context),
            patch("jarv.headsup.load_history", return_value=[]),
        ):
            app._run_slash("/new", [])

        screen = self._rendered_text(app, test_console, output, height=24)
        self.assertNotIn("Already on a new session.", screen)
        self.assertNotIn("Heads-up mode. Type /help for commands.", screen)
        self.assertIn("\u2588", screen)
        self.assertFalse(app._idle_anim_stop.is_set())

    def test_undo_syncs_visible_transcript_from_updated_history(self):
        app, _test_console, _output = self._app()
        context = SimpleNamespace(
            session_id="current-session",
            history_file=Path("history-current.json"),
        )
        app.session_context = context
        app.usage_path = Path("usage-current.json")
        app.add_user_message("first")
        app.upsert_assistant_message(None, "one")
        app.add_user_message("second")
        app.upsert_assistant_message(None, "two")

        updated_history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
        ]

        def handle_slash(command, rest, config, client, args, hint):
            app.console.print("Unsent 'second'")
            return config, client

        app.handle_slash = handle_slash

        with (
            patch("jarv.headsup.prepare_session_context", return_value=context),
            patch("jarv.headsup.load_history", return_value=updated_history),
        ):
            app._run_slash("/undo", [])

        rendered = self._entry_text(app)
        self.assertIn("first", rendered)
        self.assertIn("one", rendered)
        self.assertIn("Unsent 'second'", rendered)
        self.assertFalse(
            any(
                entry.kind == "user" and "second" in getattr(entry.renderable, "plain", "")
                for entry in app.entries
            )
        )
        self.assertNotIn("two", rendered)

    def test_redo_syncs_visible_transcript_from_updated_history(self):
        app, _test_console, _output = self._app()
        context = SimpleNamespace(
            session_id="current-session",
            history_file=Path("history-current.json"),
        )
        app.session_context = context
        app.usage_path = Path("usage-current.json")
        app.add_user_message("first")
        app.upsert_assistant_message(None, "one")

        updated_history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "two"},
        ]

        def handle_slash(command, rest, config, client, args, hint):
            app.console.print("Restored 'second'")
            return config, client

        app.handle_slash = handle_slash

        with (
            patch("jarv.headsup.prepare_session_context", return_value=context),
            patch("jarv.headsup.load_history", return_value=updated_history),
        ):
            app._run_slash("/redo", [])

        rendered = self._entry_text(app)
        self.assertIn("first", rendered)
        self.assertIn("one", rendered)
        self.assertIn("second", rendered)
        self.assertIn("two", rendered)
        self.assertIn("Restored 'second'", rendered)


class HeadsupGoldenFrameTests(unittest.TestCase):
    """Frame-level invariants at fixed terminal sizes.

    These guard the recurring WSL/ConPTY visual bugs (flush right edge, stale
    right edge, panel height) across small/medium/large terminals so a geometry
    regression fails here regardless of which size it shows up at first.
    """

    SIZES = [(40, 20), (80, 24), (120, 40), (200, 50)]

    def _render_lines(self, width, height):
        ready = threading.Event()
        ready.set()
        test_console, output = make_console(width=width)
        app = HeadsupApp(
            {"provider": "openai", "model": "test-model", "reasoning_effort": "high"},
            client=object(),
            args=None,
            agent_loader=({"module": SimpleNamespace()}, ready),
            handle_slash=lambda command, rest, config, client, args, hint: (config, client),
            maybe_command=lambda _first, _rest: None,
            render_console=test_console,
        )
        # Populate the transcript so the footer (not the intro animation) renders.
        app.add_user_message("hello from the golden frame harness")
        app.upsert_assistant_message(None, "a reply body that may need to wrap on narrow widths")
        with patch("jarv.headsup.terminal_size", return_value=(width, height)):
            test_console.print(app.render())
        return output.getvalue().splitlines()

    def test_frame_spans_terminal_width_at_any_size(self):
        for width, height in self.SIZES:
            with self.subTest(width=width, height=height):
                lines = self._render_lines(width, height)
                widest = max(cell_len(line) for line in lines)
                # Flush to the edge: never past the terminal, and reaching it.
                self.assertLessEqual(widest, width)
                self.assertEqual(widest, width)

    def test_panel_fills_terminal_height_at_any_size(self):
        for width, height in self.SIZES:
            with self.subTest(width=width, height=height):
                lines = self._render_lines(width, height)
                self.assertEqual(len(lines), height)

    def test_frame_has_rounded_borders_and_title_at_any_size(self):
        for width, height in self.SIZES:
            with self.subTest(width=width, height=height):
                rendered = "\n".join(self._render_lines(width, height))
                # Rounded box corners on the outer panel.
                self.assertIn("╭", rendered)  # top-left
                self.assertIn("╯", rendered)  # bottom-right
                self.assertIn("jarv", rendered)
                self.assertIn("Enter send", rendered)


if __name__ == "__main__":
    unittest.main()
