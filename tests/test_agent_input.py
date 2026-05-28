import io
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarv.agent import (
    build_input,
    response_start_status,
    response_wait_label,
    run_agent,
    to_response_input_item,
)
from jarv.config import DEFAULT_CONFIG
from jarv.history import SessionContext, load_history, save_history
from jarv.provider import StreamDone, TextDelta


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
                run_agent("new prompt", config, client=None)

            saved = load_history(history_file)

        self.assertEqual(len(saved), len(existing) + 2)
        self.assertEqual([item["content"] for item in saved if "content" in item], [
            "one",
            "two",
            "three",
            "new prompt",
            "four",
        ])

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
                run_agent("hello!", DEFAULT_CONFIG, client=None)

        self.assertEqual(captured["kwargs"]["prompt_cache_key"], "jarv:session-id")

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
                run_agent("hello!", DEFAULT_CONFIG, client=None)

            saved = load_history(history_file)

        self.assertEqual(saved[-1]["role"], "assistant")
        self.assertEqual(saved[-1]["content"], "recovered answer")


if __name__ == "__main__":
    unittest.main()
