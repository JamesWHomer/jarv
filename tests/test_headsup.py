import io
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console
from rich.text import Text

from jarv.agent import thought_complete_indicator, tool_complete_indicator
from jarv.agent_ui import RunningCommandCard
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.command_input import TextInput
from jarv.headsup import HeadsupAgentUI, HeadsupApp
from jarv.text_editor import initialize_text_editor


@contextmanager
def noop_context(*_args, **_kwargs):
    yield


class HeadsupTests(unittest.TestCase):
    def _app(self, *, width=50, args=None, config=None):
        ready = threading.Event()
        ready.set()
        output = io.StringIO()
        test_console = Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=width,
        )
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
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

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

    def test_prompt_box_styles_typed_text_white(self):
        app, _test_console, _output = self._app()
        initialize_text_editor(app.editor, "typed")

        lines = app._prompt_lines(40, max_lines=3)
        spans = [(span.start, span.end, str(span.style)) for span in lines[1].spans]
        self.assertFalse(str(lines[1].style))
        self.assertIn((0, 1, "dim cyan"), spans)
        self.assertIn((2, 7, "white"), spans)
        self.assertIn((39, 40, "dim cyan"), spans)

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

        self.assertIsNone(app._esc_listener_thread)
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
        refresh_count = 0

        class FakeLive:
            def refresh(self):
                nonlocal refresh_count
                refresh_count += 1

        app.live = FakeLive()

        ui.show_tool_card(
            RunningCommandCard(
                "Start-Sleep -Seconds 10",
                "",
                "fullscreen",
                time.perf_counter(),
            )
        )
        refreshes_after_show = refresh_count

        self.assertTrue(ui._refresh_wait_statuses())
        self.assertGreater(refresh_count, refreshes_after_show)

        ui.show_tool_card(Text("done"))
        refreshes_after_done = refresh_count

        self.assertFalse(ui._refresh_wait_statuses())
        self.assertEqual(refresh_count, refreshes_after_done)

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
            patch("jarv.headsup.refresh_on_resize", noop_context),
            patch("jarv.headsup.disable_mouse_capture") as disable_mouse_capture,
            patch("jarv.headsup._read_key_with_repeats", side_effect=lambda **kwargs: keys.pop(0)),
        ):
            app.run()

        self.assertEqual(disable_mouse_capture.call_count, 2)
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

    def test_prompt_accepts_multiline_paste_and_renders_rows(self):
        app, _test_console, _output = self._app()

        with patch("jarv.headsup.terminal_size", return_value=(80, 24)):
            changed, user_text = app._apply_editor_key(TextInput("first\nsecond"), 1)

        self.assertTrue(changed)
        self.assertTrue(user_text)
        self.assertEqual(app.editor["buffer"], "first\nsecond")

        lines = app._prompt_lines(40, max_lines=4)
        self.assertEqual(lines[0].plain, "\u256d" + "\u2500" * 38 + "\u256e")
        self.assertEqual(lines[1].plain, "\u2502 first" + " " * 32 + "\u2502")
        self.assertEqual(lines[2].plain, "\u2502 second " + " " * 30 + "\u2502")
        self.assertEqual(lines[3].plain, "\u2570" + "\u2500" * 38 + "\u256f")

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
            patch("jarv.headsup.refresh_on_resize", noop_context),
            patch("jarv.headsup.disable_mouse_capture") as disable_mouse_capture,
            patch("jarv.headsup._read_key_with_repeats", side_effect=lambda **kwargs: keys.pop(0)),
        ):
            app.run()

        self.assertEqual(disable_mouse_capture.call_count, 2)
        self.assertEqual(app.scroll_offset, 6)
        self.assertIsNone(app._prompt_history_index)

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

    def test_bind_cancel_token_esc_cancels(self):
        app, _test_console, _output = self._app()
        token = CancellationToken()
        keys = ["ESC"]

        with (
            patch("jarv.headsup._key_available", side_effect=[True, False]),
            patch("jarv.headsup._read_key", side_effect=lambda text_mode=False: keys.pop(0)),
        ):
            app.bind_cancel_token(token)
            if app._esc_listener_thread is not None:
                app._esc_listener_thread.join(timeout=1.0)

        self.assertTrue(token.cancelled)
        app.unbind_cancel_token()

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
        self.assertEqual(alt_screen_calls, [True, True])

    def test_all_supported_slash_commands_route_cleanly_in_headsup(self):
        cases = [
            ("/setup", ["provider"], True),
            ("/help", [], True),
            ("/about", [], True),
            ("/update", [], False),
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

    def test_new_restarts_idle_animation_thread_in_active_headsup(self):
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
                self.assertTrue(
                    self._wait_for(
                        lambda: app._idle_anim_thread is not None
                        and app._idle_anim_thread.is_alive()
                    )
                )
                self.assertFalse(app._idle_anim_stop.is_set())
            finally:
                app._idle_anim_stop.set()
                thread = app._idle_anim_thread
                if thread is not None:
                    thread.join(timeout=0.5)
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


if __name__ == "__main__":
    unittest.main()
