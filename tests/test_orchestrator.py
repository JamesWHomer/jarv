import unittest
import threading
import time
from unittest.mock import patch

from jarv.artifacts import ArtifactStore
from jarv.config import DEFAULT_CONFIG
from jarv.orchestrator import AgentNode, run_subagent_loop, spawn_batch
from jarv.provider import StreamDone, TextDelta, ToolCallDone
from jarv.cancellation import CancellationToken, TurnCancelled


class OrchestratorTests(unittest.TestCase):
    def test_subagent_passes_session_prompt_cache_key(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
            session_id="session-id",
        )
        captured = {}

        def fake_stream_response(*_args, **kwargs):
            captured["kwargs"] = kwargs
            yield ToolCallDone(
                id="fc_1",
                call_id="call_1",
                name="finish",
                arguments='{"longform": "done", "tldr": "done"}',
            )
            yield StreamDone(response=None)

        with patch("jarv.orchestrator.stream_response", side_effect=fake_stream_response):
            longform, tldr = run_subagent_loop(
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertEqual((longform, tldr), ("done", "done"))
        self.assertEqual(captured["kwargs"]["prompt_cache_key"], "jarv:session-id")

    def test_subagent_retries_once_when_finish_not_called(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )
        calls = []

        def fake_stream_response(*_args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                yield TextDelta("plain text only")
                yield StreamDone(response=None)
                return
            yield ToolCallDone(
                id="fc_1",
                call_id="call_1",
                name="finish",
                arguments='{"longform": "done", "tldr": "done"}',
            )
            yield StreamDone(response=None)

        with patch("jarv.orchestrator.stream_response", side_effect=fake_stream_response):
            longform, tldr = run_subagent_loop(
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertEqual((longform, tldr), ("done", "done"))
        self.assertEqual(len(calls), 2)

    def test_subagent_fails_after_finish_nudge_retry_exhausted(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )

        def fake_stream_response(*_args, **_kwargs):
            yield TextDelta("still no tool call")
            yield StreamDone(response=None)

        with patch("jarv.orchestrator.stream_response", side_effect=fake_stream_response):
            longform, reason = run_subagent_loop(
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertIsNone(longform)
        self.assertEqual(reason, "subagent terminated without calling finish")

    def test_subagent_uses_higher_max_tokens_for_anthropic(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )
        captured = {}

        def fake_stream_response(*_args, **kwargs):
            captured["kwargs"] = kwargs
            yield ToolCallDone(
                id="fc_1",
                call_id="call_1",
                name="finish",
                arguments='{"longform": "done", "tldr": "done"}',
            )
            yield StreamDone(response=None)

        config = dict(DEFAULT_CONFIG)
        config["provider"] = "anthropic"
        with patch("jarv.orchestrator.stream_response", side_effect=fake_stream_response):
            run_subagent_loop(node, ArtifactStore(), client=None, config=config)

        self.assertEqual(captured["kwargs"]["max_tokens"], 16384)

    def test_subagent_retries_truncated_finish_once(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )
        calls = []

        def fake_stream_response(*_args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                yield ToolCallDone(
                    id="fc_1",
                    call_id="call_1",
                    name="finish",
                    arguments='{"longform": "partial',
                )
                yield StreamDone(response={"stop_reason": "max_tokens"})
                return
            yield ToolCallDone(
                id="fc_2",
                call_id="call_2",
                name="finish",
                arguments='{"longform": "done", "tldr": "done"}',
            )
            yield StreamDone(response={"stop_reason": "tool_use"})

        with patch("jarv.orchestrator.stream_response", side_effect=fake_stream_response):
            longform, tldr = run_subagent_loop(
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertEqual((longform, tldr), ("done", "done"))
        self.assertEqual(len(calls), 2)

    def test_spawn_batch_does_not_wait_for_workers_after_cancellation(self):
        token = CancellationToken()
        release = threading.Event()
        parent = AgentNode(
            label="root",
            depth=0,
            parent_label=None,
            task="root",
            sterile=False,
        )

        def blocked_worker(*_args, **_kwargs):
            release.wait(1)
            token.throw_if_cancelled()
            return None, "cancelled"

        timer = threading.Timer(0.02, token.cancel)
        timer.start()
        started = time.perf_counter()
        try:
            with patch("jarv.orchestrator.run_subagent_loop", side_effect=blocked_worker):
                with self.assertRaises(TurnCancelled):
                    spawn_batch(
                        parent,
                        [{"label": "child", "task": "work"}],
                        ArtifactStore(),
                        client=None,
                        config=DEFAULT_CONFIG,
                        cancellation_token=token,
                    )
        finally:
            release.set()
            timer.cancel()

        self.assertLess(time.perf_counter() - started, 0.25)


if __name__ == "__main__":
    unittest.main()
