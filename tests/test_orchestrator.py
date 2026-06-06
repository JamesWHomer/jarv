import unittest
import threading
import time
from unittest.mock import patch

from jarv.artifacts import ArtifactStore
from jarv.config import DEFAULT_CONFIG
from jarv.orchestrator import AgentNode, run_subagent_loop, spawn_batch
from jarv.provider import StreamDone, ToolCallDone
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
