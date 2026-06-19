import io
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console
from rich.text import Text

from jarv.agent import thought_complete_indicator, tool_complete_indicator
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.headsup import HeadsupAgentUI, HeadsupApp


class HeadsupTests(unittest.TestCase):
    def _app(self, *, width=50, args=None):
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
            {"provider": "openai", "model": "test-model"},
            client=object(),
            args=args,
            agent_loader=({"module": SimpleNamespace()}, ready),
            handle_slash=lambda command, rest, config, client, args, hint: (config, client),
            maybe_command=lambda _first, _rest: None,
            render_console=test_console,
        )
        return app, test_console, output

    def _entry_text(self, app: HeadsupApp) -> str:
        return "\n".join(line.plain for line in app._transcript_lines(100))

    def test_render_keeps_prompt_and_footer_in_narrow_terminal(self):
        app, test_console, output = self._app(width=42)
        app.add_user_message("hello from a narrow terminal")
        app.upsert_assistant_message(None, "reply body")

        with patch("jarv.headsup.terminal_size", return_value=(42, 12)):
            test_console.print(app.render())

        rendered = output.getvalue()
        self.assertIn("jarv", rendered)
        self.assertIn("jarv>", rendered)
        self.assertIn("\u203a ", rendered)
        self.assertIn("hello from a narrow terminal", rendered)
        self.assertIn("Enter send", rendered)
        self.assertIn("reply body", rendered)

    def test_top_title_styles_jarv_like_settings_menu(self):
        app, test_console, output = self._app(width=80)
        with patch("jarv.headsup.terminal_size", return_value=(80, 12)):
            test_console.print(app.render())

        rendered = output.getvalue()
        self.assertIn("jarv \u25b8 heads up", rendered)

    def test_render_shows_provider_model_top_right_and_bottom_usage_status(self):
        app, test_console, output = self._app(width=80)
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
        self.assertIn("jarv \u25b8 heads up", lines[0])
        self.assertIn("openai / test-model", lines[0])
        self.assertNotIn("openai / test-model", lines[-1])
        self.assertIn("cost $0.123", lines[-1])
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
        self.assertTrue(
            any(span.start <= money_index < span.end and str(span.style) == "green" for span in status.spans)
        )
        self.assertTrue(
            any(
                span.start <= percent_index < span.end and str(span.style) == "bold bright_red"
                for span in status.spans
            )
        )

    def test_usage_line_appends_to_transcript(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)
        before = len(app.entries)
        usage = Text("Usage: 1 in", style="dim")

        ui.show_usage_line(usage)

        self.assertEqual(len(app.entries), before + 1)
        self.assertEqual(app.entries[-1].kind, "usage")
        self.assertEqual(app.entries[-1].renderable.plain, "Usage: 1 in")

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

    def test_active_wait_status_refresh_updates_elapsed_text(self):
        app, _test_console, _output = self._app()
        ui = HeadsupAgentUI(app)
        ui._response_started_at = 0.0
        ui._response_waiting = True
        ui._response_status_index = app.upsert_status(None, Text("old"))

        with patch("jarv.headsup.time.perf_counter", return_value=2.0):
            self.assertTrue(ui._refresh_wait_statuses())

        self.assertIn("2s", app.entries[ui._response_status_index].renderable.plain)

        ui.complete_response_phase("Thought for 2.0 seconds.")
        completed = app.entries[ui._response_status_index].renderable.plain

        with patch("jarv.headsup.time.perf_counter", return_value=5.0):
            self.assertFalse(ui._refresh_wait_statuses())

        self.assertEqual(app.entries[ui._response_status_index].renderable.plain, completed)

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
        self.assertFalse(app._handle_prompt_dismiss())
        self.assertTrue(app._exit_armed)
        self.assertTrue(app._handle_prompt_dismiss())

    def test_prompt_dismiss_clears_draft_with_notice(self):
        app, _test_console, _output = self._app()
        app.editor["buffer"] = "draft"
        app._exit_armed = True

        self.assertFalse(app._handle_prompt_dismiss())

        self.assertEqual(app.editor["buffer"], "")
        self.assertTrue(app._exit_armed)
        notices = [entry.renderable.plain for entry in app.entries if entry.kind == "notice"]
        self.assertIn("Draft cleared.", notices)

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
                    self.assertEqual(calls[0], "stop")
                    self.assertIn(("start", True), calls)
                    self.assertTrue(
                        any(
                            getattr(entry.renderable, "plain", "").strip() == f"handled {command}"
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
        self.assertEqual(calls[:2], ["stop", ("start", True)])
        self.assertIn("command output", rendered)
        self.assertNotIn("live frame should not be captured", rendered)

    def test_new_refreshes_session_context_and_clears_visible_transcript(self):
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
        self.assertEqual(app.session_context.session_id, "new-session")
        self.assertEqual(app.usage_path, Path("usage-new.json"))
        self.assertIn("new session ready", rendered)
        self.assertNotIn("old visible message", rendered)
        self.assertNotIn("old visible reply", rendered)

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
