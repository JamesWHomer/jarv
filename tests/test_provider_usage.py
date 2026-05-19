import unittest
from types import SimpleNamespace

from jarv.provider import StreamDone, TextDelta, _stream_chat_completions


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        pass


class ProviderUsageTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
