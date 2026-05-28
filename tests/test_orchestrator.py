import unittest
from unittest.mock import patch

from jarv.artifacts import ArtifactStore
from jarv.config import DEFAULT_CONFIG
from jarv.orchestrator import AgentNode, run_subagent_loop
from jarv.provider import StreamDone, ToolCallDone


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


if __name__ == "__main__":
    unittest.main()
