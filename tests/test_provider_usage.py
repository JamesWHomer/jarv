import threading
import time
import unittest
from unittest.mock import patch

from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.provider import (
    ReasoningDone,
    ReasoningStarted,
    RetryableStreamError,
    StreamDone,
    TextDelta,
    ToolCallDone,
    ToolCallStarted,
    _stream_chat_completions,
    _stream_gemini,
    _stream_responses_api,
    response_output_text,
    responses_input_id,
    stream_response,
)


class ProviderUsageTests(unittest.TestCase):
    def test_responses_input_id_normalizes_invalid_ids(self):
        self.assertEqual(responses_input_id("fc_123", "fc"), "fc_123")
        shortened = responses_input_id("call_" + ("x" * 100), "fc")
        self.assertTrue(shortened.startswith("fc_"))
        self.assertLessEqual(len(shortened), 64)

    def test_response_output_text_supports_direct_http_dict(self):
        response = {
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }]
        }
        self.assertEqual(response_output_text(response), "hello")

    def test_responses_stream_maps_reasoning_tools_cache_key_and_usage(self):
        captured = {}

        def fake_stream(_client, payload, **_kwargs):
            captured["payload"] = payload
            yield {
                "type": "response.created",
                "response": {"id": "resp_1"},
            }
            yield {
                "type": "response.output_item.added",
                "item": {"type": "reasoning", "id": "rs_1"},
            }
            yield {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "run",
                    "arguments": "",
                },
            }
            yield {"type": "response.output_text.delta", "delta": "hi"}
            yield {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "run",
                    "arguments": "{}",
                },
            }
            yield {
                "type": "response.output_item.done",
                "item": {"type": "reasoning", "id": "rs_1", "summary": []},
            }
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 12},
                        "output_tokens": 3,
                    },
                    "output": [],
                },
            }

        with patch("jarv.openai_http.stream_response", side_effect=fake_stream):
            events = list(_stream_responses_api(
                object(),
                "model",
                "system",
                [],
                [],
                prompt_cache_key="jarv:session",
            ))

        self.assertEqual(captured["payload"]["prompt_cache_key"], "jarv:session")
        self.assertIs(captured["payload"]["store"], True)
        self.assertIsInstance(events[0], ReasoningStarted)
        self.assertIsInstance(events[1], ToolCallStarted)
        self.assertEqual(events[1].name, "run")
        self.assertIsInstance(events[2], TextDelta)
        self.assertIsInstance(events[3], ToolCallDone)
        self.assertIsInstance(events[4], ReasoningDone)
        self.assertIsInstance(events[5], StreamDone)

    def test_responses_stream_recovers_after_disconnect(self):
        def broken(_client, _payload, **_kwargs):
            yield {"type": "response.created", "response": {"id": "resp_1"}}
            yield {"type": "response.output_text.delta", "delta": "partial"}
            raise RuntimeError("disconnect")

        recovered = {
            "id": "resp_1",
            "status": "completed",
            "output_text": "partial and recovered",
            "output": [],
        }
        with (
            patch("jarv.openai_http.stream_response", side_effect=broken),
            patch("jarv.openai_http.retrieve_response", return_value=recovered),
        ):
            events = list(_stream_responses_api(
                object(), "model", "system", [], []
            ))
        self.assertEqual(response_output_text(events[-1].response), "partial and recovered")

    def test_responses_stream_polls_after_clean_eof_until_response_completes(self):
        def truncated(_client, _payload, **_kwargs):
            yield {"type": "response.created", "response": {"id": "resp_1"}}
            yield {
                "type": "response.output_text.delta",
                "response_id": "resp_1",
                "delta": "partial",
            }

        recovered = {
            "id": "resp_1",
            "status": "completed",
            "output_text": "partial and recovered",
            "output": [],
        }
        retrieve_results = [
            {"id": "resp_1", "status": "in_progress", "output": []},
            {"id": "resp_1", "status": "in_progress", "output": []},
            recovered,
        ]
        with (
            patch("jarv.openai_http.stream_response", side_effect=truncated),
            patch(
                "jarv.openai_http.retrieve_response",
                side_effect=retrieve_results,
            ) as retrieve,
            patch("jarv.provider._sleep_for_openai_recovery") as recovery_sleep,
        ):
            events = list(_stream_responses_api(
                object(), "model", "system", [], []
            ))

        self.assertEqual(retrieve.call_count, 3)
        self.assertEqual(recovery_sleep.call_count, 2)
        self.assertIsInstance(events[-1], StreamDone)
        self.assertEqual(response_output_text(events[-1].response), "partial and recovered")

    def test_responses_stream_recovers_using_top_level_response_id(self):
        def truncated(_client, _payload, **_kwargs):
            yield {
                "type": "response.output_text.delta",
                "response_id": "resp_top_level",
                "delta": "partial",
            }

        recovered = {
            "id": "resp_top_level",
            "status": "completed",
            "output_text": "partial and recovered",
            "output": [],
        }
        with (
            patch("jarv.openai_http.stream_response", side_effect=truncated),
            patch("jarv.openai_http.retrieve_response", return_value=recovered) as retrieve,
        ):
            events = list(_stream_responses_api(
                object(), "model", "system", [], []
            ))

        retrieve.assert_called_once()
        self.assertEqual(response_output_text(events[-1].response), "partial and recovered")

    def test_responses_stream_recovery_failure_preserves_retrieval_error(self):
        def truncated(_client, _payload, **_kwargs):
            yield {"type": "response.created", "response": {"id": "resp_1"}}
            yield {"type": "response.output_text.delta", "delta": "partial"}

        with (
            patch("jarv.openai_http.stream_response", side_effect=truncated),
            patch(
                "jarv.openai_http.retrieve_response",
                side_effect=RuntimeError("lookup failed"),
            ),
            patch("jarv.provider._sleep_for_openai_recovery"),
        ):
            with self.assertRaisesRegex(
                RetryableStreamError,
                "last retrieval error: lookup failed",
            ):
                list(_stream_responses_api(
                    object(), "model", "system", [], []
                ))

    def test_responses_stream_recovery_failure_preserves_last_status(self):
        def truncated(_client, _payload, **_kwargs):
            yield {"type": "response.created", "response": {"id": "resp_1"}}

        with (
            patch("jarv.openai_http.stream_response", side_effect=truncated),
            patch(
                "jarv.openai_http.retrieve_response",
                return_value={
                    "id": "resp_1",
                    "status": "in_progress",
                    "output": [],
                },
            ),
            patch("jarv.provider._sleep_for_openai_recovery"),
        ):
            with self.assertRaisesRegex(
                RetryableStreamError,
                "last recovery status: in_progress",
            ):
                list(_stream_responses_api(
                    object(), "model", "system", [], []
                ))

    def test_responses_stream_cancellation_does_not_attempt_recovery(self):
        def cancelled(_client, _payload, **_kwargs):
            yield {"type": "response.created", "response": {"id": "resp_1"}}
            raise TurnCancelled

        with (
            patch("jarv.openai_http.stream_response", side_effect=cancelled),
            patch("jarv.openai_http.retrieve_response") as retrieve,
        ):
            with self.assertRaises(TurnCancelled):
                list(_stream_responses_api(
                    object(), "model", "system", [], []
                ))

        retrieve.assert_not_called()

    def test_chat_stream_maps_reasoning_tools_and_usage(self):
        chunks = [
            {
                "id": "chat_1",
                "choices": [{
                    "delta": {"reasoning_content": "thinking"},
                    "finish_reason": None,
                }],
            },
            {
                "choices": [{
                    "delta": {
                        "content": "hi",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "run", "arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            },
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        ]
        with patch("jarv.openai_http.stream_chat", return_value=iter(chunks)):
            events = list(_stream_chat_completions(
                object(), "model", "system", [], []
            ))
        self.assertIsInstance(events[0], ReasoningStarted)
        self.assertIsInstance(events[1], TextDelta)
        self.assertIsInstance(events[2], ToolCallStarted)
        self.assertIsInstance(events[3], ToolCallDone)
        self.assertEqual(events[-1].response["usage"]["prompt_tokens"], 5)

    def test_gemini_stream_preserves_thought_and_tool_signatures(self):
        protocol_events = [
            {"type": "reasoning_started", "id": "gemini-thinking"},
            {
                "type": "reasoning_part",
                "provider_content": [{"text": "thought", "thought": True}],
            },
            {
                "type": "tool_call",
                "id": "call_1",
                "name": "run",
                "arguments": "{}",
                "provider_content": [{
                    "functionCall": {"name": "run", "args": {}},
                    "thoughtSignature": "signed",
                }],
            },
            {"type": "reasoning_done", "id": "gemini-thinking"},
            {"type": "done", "response": {"usage": {"input_tokens": 2}}},
        ]
        with patch("jarv.gemini_http.stream_content", return_value=iter(protocol_events)):
            events = list(_stream_gemini(
                object(), {"provider": "gemini"}, "model", "system", [], []
            ))
        reasoning = next(event for event in events if isinstance(event, ReasoningDone))
        tool_start = next(event for event in events if isinstance(event, ToolCallStarted))
        tool = next(event for event in events if isinstance(event, ToolCallDone))
        self.assertTrue(reasoning.provider_content[0]["thought"])
        self.assertEqual(tool_start.name, "run")
        self.assertEqual(tool.provider_content[0]["thoughtSignature"], "signed")

    def test_windows_stream_bridge_observes_cancellation_promptly(self):
        token = CancellationToken()

        def blocking_direct(*_args, **_kwargs):
            while True:
                token.throw_if_cancelled()
                time.sleep(0.005)
                if False:
                    yield None

        timer = threading.Timer(0.02, token.cancel)
        timer.start()
        started = time.perf_counter()
        try:
            with (
                patch("jarv.provider.sys.platform", "win32"),
                patch("jarv.provider._stream_response_direct", side_effect=blocking_direct),
            ):
                with self.assertRaises(TurnCancelled):
                    list(stream_response(
                        object(), {}, "model", "system", [], [],
                        cancellation_token=token,
                    ))
        finally:
            timer.cancel()
        self.assertLess(time.perf_counter() - started, 0.25)

    def test_posix_streaming_stays_on_calling_thread(self):
        caller_thread = threading.get_ident()
        producer_threads = []

        def direct(*_args, **_kwargs):
            producer_threads.append(threading.get_ident())
            yield TextDelta("ok")

        with (
            patch("jarv.provider.sys.platform", "linux"),
            patch("jarv.provider._stream_response_direct", side_effect=direct),
        ):
            events = list(stream_response(object(), {}, "model", "system", [], []))
        self.assertEqual(events[0].delta, "ok")
        self.assertEqual(producer_threads, [caller_thread])


if __name__ == "__main__":
    unittest.main()
