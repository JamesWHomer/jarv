import unittest
from types import SimpleNamespace

from jarv.provider import StreamDone, TextDelta, _stream_chat_completions, responses_input_id


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        pass


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


if __name__ == "__main__":
    unittest.main()
