import unittest
from unittest.mock import patch
from types import SimpleNamespace

from jarv.provider import (
    ReasoningDone,
    ReasoningStarted,
    StreamDone,
    TextDelta,
    _stream_chat_completions,
    _stream_litellm,
    _stream_responses_api,
    response_output_text,
    responses_input_id,
)


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        pass


class FakeResponseStream(FakeStream):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def get_final_response(self):
        return SimpleNamespace(usage=None)


class BrokenResponseStream(FakeResponseStream):
    def __iter__(self):
        yield SimpleNamespace(
            type="response.created",
            response=SimpleNamespace(id="resp_123"),
        )
        yield SimpleNamespace(type="response.output_text.delta", delta="partial")
        raise RuntimeError("incomplete chunked read")


class ProviderUsageTests(unittest.TestCase):
    def test_responses_input_id_keeps_valid_id(self):
        self.assertEqual(responses_input_id("fc_123", "fc"), "fc_123")

    def test_responses_input_id_shortens_overlong_id(self):
        item_id = "fc_" + ("x" * 100)

        result = responses_input_id(item_id, "fc")

        self.assertLessEqual(len(result), 64)
        self.assertTrue(result.startswith("fc_"))
        self.assertEqual(result, responses_input_id(item_id, "fc"))
        self.assertNotEqual(result, item_id)

    def test_responses_input_id_replaces_wrong_prefix(self):
        result = responses_input_id("call_7119a55952524247b01522fc", "fc")

        self.assertLessEqual(len(result), 64)
        self.assertTrue(result.startswith("fc_"))

    def test_chat_stream_keeps_usage_when_final_chunk_has_choices(self):
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hi", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=usage,
            ),
        ]

        stream = FakeStream(chunks)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: stream)
            )
        )

        events = list(_stream_chat_completions(client, "model", "system", [], []))

        self.assertIsInstance(events[0], TextDelta)
        self.assertIsInstance(events[-1], StreamDone)
        self.assertIs(events[-1].response, chunks[-1])

    def test_chat_stream_emits_reasoning_started_from_reasoning_content(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(reasoning_content="thinking", content=None, tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hi", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: FakeStream(chunks))
            )
        )

        events = list(_stream_chat_completions(client, "model", "system", [], []))

        self.assertIsInstance(events[0], ReasoningStarted)
        self.assertIsInstance(events[1], TextDelta)

    def test_chat_stream_emits_reasoning_started_from_extra_reasoning(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            additional_kwargs={"reasoning": "thinking"},
                            content=None,
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hi", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: FakeStream(chunks))
            )
        )

        events = list(_stream_chat_completions(client, "model", "system", [], []))

        self.assertIsInstance(events[0], ReasoningStarted)
        self.assertIsInstance(events[1], TextDelta)

    def test_chat_stream_emits_reasoning_started_from_content_block(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=[{"type": "thinking", "thinking": "thinking"}],
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: FakeStream(chunks))
            )
        )

        events = list(_stream_chat_completions(client, "model", "system", [], []))

        self.assertIsInstance(events[0], ReasoningStarted)

    def test_chat_stream_sanitizes_lone_surrogates_before_api_call(self):
        captured = {}
        stream = FakeStream([])

        def create(**kwargs):
            captured["kwargs"] = kwargs
            return stream

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=create
                )
            )
        )

        list(_stream_chat_completions(
            client,
            "model",
            "system\udc8f",
            [],
            [{"role": "user", "content": "abc\udc8fdef"}],
        ))

        messages = captured["kwargs"]["messages"]
        self.assertEqual(messages[0]["content"], "system?")
        self.assertEqual(messages[1]["content"], "abc?def")
        str(captured["kwargs"]).encode("utf-8")

    def test_responses_stream_emits_reasoning_started_before_text(self):
        stream = FakeResponseStream([
            SimpleNamespace(
                type="response.output_item.added",
                item=SimpleNamespace(type="reasoning", id="rs_123"),
            ),
            SimpleNamespace(type="response.output_text.delta", delta="hi"),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(type="reasoning", id="rs_123", summary=[]),
            ),
        ])
        client = SimpleNamespace(
            responses=SimpleNamespace(stream=lambda **_kwargs: stream)
        )

        events = list(_stream_responses_api(client, "model", "system", [], []))

        self.assertIsInstance(events[0], ReasoningStarted)
        self.assertEqual(events[0].id, "rs_123")
        self.assertIsInstance(events[1], TextDelta)
        self.assertIsInstance(events[2], ReasoningDone)
        self.assertIsInstance(events[-1], StreamDone)

    def test_responses_stream_emits_reasoning_started_from_reasoning_text_delta(self):
        stream = FakeResponseStream([
            SimpleNamespace(type="response.reasoning_text.delta", delta="thinking"),
            SimpleNamespace(type="response.output_text.delta", delta="hi"),
        ])
        client = SimpleNamespace(
            responses=SimpleNamespace(stream=lambda **_kwargs: stream)
        )

        events = list(_stream_responses_api(client, "model", "system", [], []))

        self.assertIsInstance(events[0], ReasoningStarted)
        self.assertIsInstance(events[1], TextDelta)

    def test_responses_stream_forwards_prompt_cache_key(self):
        captured = {}
        stream = FakeResponseStream([])

        def response_stream(**kwargs):
            captured["kwargs"] = kwargs
            return stream

        client = SimpleNamespace(
            responses=SimpleNamespace(stream=response_stream)
        )

        list(_stream_responses_api(
            client,
            "model",
            "system",
            [],
            [],
            prompt_cache_key="jarv:session-id",
        ))

        self.assertEqual(captured["kwargs"]["prompt_cache_key"], "jarv:session-id")

    def test_responses_stream_recovers_completed_response_after_disconnect(self):
        recovered = SimpleNamespace(
            id="resp_123",
            status="completed",
            output_text="partial and recovered",
            output=[],
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(
                stream=lambda **_kwargs: BrokenResponseStream([]),
                retrieve=lambda response_id: recovered if response_id == "resp_123" else None,
            )
        )

        events = list(_stream_responses_api(client, "model", "system", [], []))

        self.assertIsInstance(events[0], TextDelta)
        self.assertIsInstance(events[-1], StreamDone)
        self.assertIs(events[-1].response, recovered)
        self.assertEqual(response_output_text(events[-1].response), "partial and recovered")

    def test_litellm_stream_emits_reasoning_started_from_thinking_blocks(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(thinking_blocks=[{"type": "thinking"}], content=None, tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hi", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]
        fake_litellm = SimpleNamespace(completion=lambda **_kwargs: FakeStream(chunks))

        with patch("jarv.litellm_compat.import_litellm", return_value=fake_litellm):
            events = list(_stream_litellm({"provider": "anthropic"}, "model", "system", [], []))

        self.assertIsInstance(events[0], ReasoningStarted)
        self.assertIsInstance(events[1], TextDelta)

    def test_litellm_stream_emits_reasoning_started_from_provider_specific_reasoning(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=None,
                            provider_specific_fields={
                                "reasoningContent": {"type": "thinking_delta", "text": "thinking"}
                            },
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hi", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]
        fake_litellm = SimpleNamespace(completion=lambda **_kwargs: FakeStream(chunks))

        with patch("jarv.litellm_compat.import_litellm", return_value=fake_litellm):
            events = list(_stream_litellm({"provider": "anthropic"}, "model", "system", [], []))

        self.assertIsInstance(events[0], ReasoningStarted)
        self.assertIsInstance(events[1], TextDelta)

    def test_litellm_stream_forwards_reasoning_effort(self):
        captured = {}
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hi", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]

        def completion(**kwargs):
            captured["kwargs"] = kwargs
            return FakeStream(chunks)

        fake_litellm = SimpleNamespace(completion=completion)

        with patch("jarv.litellm_compat.import_litellm", return_value=fake_litellm):
            list(_stream_litellm(
                {"provider": "anthropic"},
                "model",
                "system",
                [],
                [],
                reasoning={"effort": "high"},
            ))

        self.assertEqual(captured["kwargs"]["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
