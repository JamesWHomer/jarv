import io
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console
from jarv.agent import (
    _dispatch_ask_user,
    build_input,
    _format_agent_usage_line,
    response_start_status,
    response_wait_label,
    run_agent,
    to_response_input_item,
)
from jarv.config import DEFAULT_CONFIG
from jarv.history import SessionContext, load_history, save_history
from jarv.history import redo_file_for
from jarv.provider import StreamDone, TextDelta, ToolCallDone


class AgentInputTests(unittest.TestCase):
    def test_response_wait_label_is_neutral_without_reasoning(self):
        self.assertEqual(response_wait_label(has_reasoning=False), "Waiting")

    def test_response_wait_label_uses_thinking_with_reasoning(self):
        self.assertEqual(response_wait_label(has_reasoning=True), "Thinking")

    def test_response_start_status_uses_reasoning_label_for_reasoning_events(self):
        self.assertEqual(
            response_start_status(4.34, has_reasoning=True),
            "Thought for 4.3 seconds.",
        )

    def test_response_start_status_uses_first_token_label_without_reasoning(self):
        self.assertEqual(
            response_start_status(1.0, has_reasoning=False),
            "Started responding in 1.0 second.",
        )

    def test_function_call_id_is_shortened_for_responses_input(self):
        item = {
            "type": "function_call",
            "id": "fc_" + ("x" * 100),
            "call_id": "call_123",
            "name": "run_command",
            "arguments": "{}",
        }

        api_item = to_response_input_item(item)

        self.assertLessEqual(len(api_item["id"]), 64)
        self.assertTrue(api_item["id"].startswith("fc_"))
        self.assertEqual(api_item["call_id"], "call_123")

    def test_function_call_id_gets_responses_prefix(self):
        item = {
            "type": "function_call",
            "id": "call_7119a55952524247b01522fc",
            "call_id": "call_7119a55952524247b01522fc",
            "name": "run_command",
            "arguments": "{}",
        }

        api_item = to_response_input_item(item)

        self.assertLessEqual(len(api_item["id"]), 64)
        self.assertTrue(api_item["id"].startswith("fc_"))
        self.assertEqual(api_item["call_id"], "call_7119a55952524247b01522fc")

    def test_function_call_preserves_provider_content_for_gemini_replay(self):
        provider_content = [{
            "functionCall": {"name": "run_command", "args": {}},
            "thoughtSignature": "signed",
        }]
        api_item = to_response_input_item({
            "type": "function_call",
            "id": "call_1",
            "call_id": "call_1",
            "name": "run_command",
            "arguments": "{}",
            "provider_content": provider_content,
        })
        self.assertEqual(api_item["provider_content"], provider_content)

    def test_reasoning_id_is_shortened_for_responses_input(self):
        item = {
            "type": "reasoning",
            "id": "rs_" + ("x" * 100),
            "summary": [],
        }

        api_item = to_response_input_item(item)

        self.assertLessEqual(len(api_item["id"]), 64)
        self.assertTrue(api_item["id"].startswith("rs_"))

    def test_build_input_limits_context_without_mutating_history(self):
        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "newer answer"},
        ]
        original = [dict(item) for item in history]

        api_items = build_input(history, max_history=2)

        self.assertEqual(
            api_items,
            [
                {"role": "user", "content": "new"},
                {"role": "assistant", "content": "newer answer"},
            ],
        )
        self.assertEqual(history, original)

    def test_build_input_counts_tool_call_and_output_as_history_items(self):
        history = [
            {"role": "user", "content": "run a command"},
            {
                "type": "function_call",
                "id": "call_1",
                "call_id": "call_1",
                "name": "run_command",
                "arguments": '{"command": "pwd"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "C:\\work",
            },
            {"role": "assistant", "content": "done"},
        ]

        self.assertEqual(len(build_input(history, max_history=4)), 4)
        self.assertEqual(build_input(history, max_history=3), [])

    def test_run_agent_persists_full_history_even_when_context_limit_is_lower(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            existing = [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ]
            save_history(existing, history_file)
            context = SessionContext(
                session_id="session-id",
                session_label="test session",
                history_file=history_file,
                now=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )
            config = {**DEFAULT_CONFIG, "max_history": 2}

            def fake_stream_response(*_args, **_kwargs):
                yield TextDelta("four")
                yield StreamDone(response=None)

            with (
                patch("jarv.agent.prepare_session_context", return_value=context),
                patch("jarv.agent.stream_response", side_effect=fake_stream_response),
                patch("jarv.agent.sys.stdout", new=io.StringIO()),
            ):
                run_agent("new prompt", config, client=object())

            saved = load_history(history_file)

        self.assertEqual(len(saved), len(existing) + 2)
        self.assertEqual([item["content"] for item in saved if "content" in item], [
            "one",
            "two",
            "three",
            "new prompt",
            "four",
        ])

    def test_run_agent_cancellation_checkpoints_turn_and_clears_redo(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            existing = [
                {"role": "user", "content": "old prompt"},
                {"role": "assistant", "content": "old answer"},
            ]
            save_history(existing, history_file)
            redo_path = redo_file_for(history_file)
            redo_path.write_text('[[{"role":"user","content":"redo me"}]]', encoding="utf-8")
            artifact_path = history_file.with_name("artifacts.json")
            artifact_path.write_text(
                '{"existing":{"longform":"kept","tldr":"kept","owner_label":"old"}}',
                encoding="utf-8",
            )
            artifact_before = artifact_path.read_text(encoding="utf-8")
            context = SessionContext(
                session_id="session-id",
                session_label="test session",
                history_file=history_file,
                now=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )

            def interrupted_stream(*_args, **_kwargs):
                yield TextDelta("partial")
                raise KeyboardInterrupt

            with (
                patch("jarv.agent.prepare_session_context", return_value=context),
                patch("jarv.agent.stream_response", side_effect=interrupted_stream),
                patch("jarv.agent.sys.stdout", new=io.StringIO()),
            ):
                result = run_agent("edit this prompt", DEFAULT_CONFIG, client=object())

            saved = load_history(history_file)
            redo_exists = redo_path.exists()
            artifact_after = artifact_path.read_text(encoding="utf-8")

        self.assertTrue(result.cancelled)
        self.assertEqual(result.prompt, "edit this prompt")
        self.assertEqual(saved[:2], existing)
        self.assertEqual(saved[2]["content"], "edit this prompt")
        self.assertEqual(saved[3]["content"], "partial")
        self.assertEqual(saved[4]["content"], "[Turn cancelled by user.]")
        self.assertFalse(redo_exists)
        self.assertEqual(artifact_after, artifact_before)

    def test_run_agent_cancellation_records_interrupted_and_pending_tools(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            context = SessionContext(
                session_id="session-id",
                session_label="test session",
                history_file=history_file,
                now=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )

            def tool_stream(*_args, **_kwargs):
                yield ToolCallDone(
                    id="fc_active",
                    call_id="call_active",
                    name="run_command",
                    arguments='{"command":"change-files"}',
                )
                yield ToolCallDone(
                    id="fc_pending",
                    call_id="call_pending",
                    name="run_command",
                    arguments='{"command":"more-changes"}',
                )
                yield StreamDone(response=None)

            with (
                patch("jarv.agent.prepare_session_context", return_value=context),
                patch("jarv.agent.stream_response", side_effect=tool_stream),
                patch(
                    "jarv.agent._dispatch_run_command_with_ui",
                    side_effect=KeyboardInterrupt,
                ),
                patch("jarv.agent.sys.stdout", new=io.StringIO()),
            ):
                result = run_agent("change the files", DEFAULT_CONFIG, client=object())

            saved = load_history(history_file)

        self.assertTrue(result.cancelled)
        self.assertEqual(saved[0]["content"], "change the files")
        self.assertEqual(saved[1]["call_id"], "call_active")
        self.assertIn("may have made partial changes", saved[2]["output"])
        self.assertEqual(saved[3]["call_id"], "call_pending")
        self.assertIn("before execution", saved[4]["output"])
        self.assertEqual(saved[5]["content"], "[Turn cancelled by user.]")

    def test_ask_user_ctrl_c_propagates_to_cancel_turn(self):
        with patch("jarv.agent._read_user_input", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                _dispatch_ask_user({"question": "Continue?"})

    def test_run_agent_passes_session_prompt_cache_key(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            context = SessionContext(
                session_id="session-id",
                session_label="test session",
                history_file=history_file,
                now=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )
            captured = {}

            def fake_stream_response(*_args, **kwargs):
                captured["kwargs"] = kwargs
                yield TextDelta("hello")
                yield StreamDone(response=None)

            with (
                patch("jarv.agent.prepare_session_context", return_value=context),
                patch("jarv.agent.stream_response", side_effect=fake_stream_response),
                patch("jarv.agent.sys.stdout", new=io.StringIO()),
            ):
                run_agent("hello!", DEFAULT_CONFIG, client=object())

        self.assertEqual(captured["kwargs"]["prompt_cache_key"], "jarv:session-id")

    def test_run_agent_starts_wait_indicator_before_lazy_client_creation(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            context = SessionContext(
                session_id="session-id",
                session_label="test session",
                history_file=history_file,
                now=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )
            events = []

            class FakeLive:
                def __init__(self, *_args, **_kwargs):
                    pass

                def start(self):
                    events.append("live_start")

                def stop(self):
                    events.append("live_stop")

                def update(self, *_args, **_kwargs):
                    pass

            class TtyStringIO(io.StringIO):
                def isatty(self):
                    return True

            def fake_create_client(_config):
                events.append("create_client")
                return object()

            def fake_stream_response(*_args, **_kwargs):
                events.append("stream_response")
                yield TextDelta("hello")
                yield StreamDone(response=None)

            with (
                patch("jarv.agent.prepare_session_context", return_value=context),
                patch("jarv.agent.create_client", side_effect=fake_create_client),
                patch("jarv.agent.stream_response", side_effect=fake_stream_response),
                patch("jarv.agent.sys.stdout", new=TtyStringIO()),
                patch("jarv.agent.Live", FakeLive),
            ):
                run_agent("hello!", DEFAULT_CONFIG, client=None, incognito=True)

        self.assertEqual(events[0], "live_start")
        self.assertLess(events.index("live_start"), events.index("create_client"))
        self.assertLess(events.index("create_client"), events.index("stream_response"))

    def test_run_agent_persists_text_recovered_from_final_response(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            context = SessionContext(
                session_id="session-id",
                session_label="test session",
                history_file=history_file,
                now=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )
            recovered_response = type(
                "Response",
                (),
                {"output_text": "recovered answer", "status": "completed"},
            )()

            def fake_stream_response(*_args, **_kwargs):
                yield StreamDone(response=recovered_response)

            with (
                patch("jarv.agent.prepare_session_context", return_value=context),
                patch("jarv.agent.stream_response", side_effect=fake_stream_response),
                patch("jarv.agent.sys.stdout", new=io.StringIO()),
            ):
                run_agent("hello!", DEFAULT_CONFIG, client=object())

            saved = load_history(history_file)

        self.assertEqual(saved[-1]["role"], "assistant")
        self.assertEqual(saved[-1]["content"], "recovered answer")

    def test_run_agent_does_not_print_usage_by_default(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            context = SessionContext(
                session_id="session-id",
                session_label="test session",
                history_file=history_file,
                now=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )
            response = SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=1200,
                    cached_input_tokens=200,
                    output_tokens=300,
                    total_tokens=1500,
                )
            )
            console_output = io.StringIO()

            def fake_stream_response(*_args, **_kwargs):
                yield TextDelta("hello")
                yield StreamDone(response=response)

            with (
                patch("jarv.agent.prepare_session_context", return_value=context),
                patch("jarv.agent.stream_response", side_effect=fake_stream_response),
                patch("jarv.agent.sys.stdout", new=io.StringIO()),
                patch("jarv.agent.console", new=Console(file=console_output, force_terminal=False, color_system=None)),
            ):
                run_agent("hello!", DEFAULT_CONFIG, client=object())

        self.assertNotIn("Usage:", console_output.getvalue())

    def test_run_agent_prints_usage_when_enabled(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"
            context = SessionContext(
                session_id="session-id",
                session_label="test session",
                history_file=history_file,
                now=datetime(2026, 5, 21, tzinfo=timezone.utc),
            )
            response = SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=1200,
                    cached_input_tokens=200,
                    output_tokens=300,
                    total_tokens=1500,
                )
            )
            console_output = io.StringIO()

            def fake_stream_response(*_args, **_kwargs):
                yield TextDelta("hello")
                yield StreamDone(response=response)

            with (
                patch("jarv.agent.prepare_session_context", return_value=context),
                patch("jarv.agent.stream_response", side_effect=fake_stream_response),
                patch("jarv.agent.sys.stdout", new=io.StringIO()),
                patch("jarv.agent.console", new=Console(file=console_output, force_terminal=False, color_system=None)),
            ):
                run_agent(
                    "hello!",
                    {**DEFAULT_CONFIG, "print_usage_after_agent": True},
                    client=object(),
                )

        output = console_output.getvalue()
        self.assertIn("Usage:", output)
        self.assertIn("1,200 in (200 cached)", output)
        self.assertIn("300 out", output)
        self.assertIn("1,500 last", output)
        self.assertIn("1,500 session", output)

    def test_format_agent_usage_line_marks_estimated_usage(self):
        line = _format_agent_usage_line(
            {
                "totals": {
                    "total_tokens": 2345,
                    "estimated_cost_usd": 0.031,
                },
                "last_root_request": {
                    "input_tokens": 1000,
                    "cached_input_tokens": 0,
                    "output_tokens": 111,
                    "total_tokens": 1111,
                    "estimated": True,
                },
            }
        )

        self.assertIsNotNone(line)
        plain = line.plain
        self.assertIn("1,000 in", plain)
        self.assertIn("111 out", plain)
        self.assertIn("1,111 last", plain)
        self.assertIn("2,345 session", plain)
        self.assertIn("est. $0.03", plain)
        self.assertIn("usage estimated", plain)


if __name__ == "__main__":
    unittest.main()
