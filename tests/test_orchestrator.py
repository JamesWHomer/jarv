import unittest
import threading
import time
from unittest.mock import patch

from jarv.artifacts import ArtifactStore
from jarv.config import DEFAULT_CONFIG
from jarv.orchestrator import (
    RUN_COMMAND_TOOL,
    AgentNode,
    dispatch_tool,
    run_subagent_loop,
    spawn_batch,
)
from jarv.provider import (
    RetryableStreamError,
    StreamDone,
    TextDelta,
    ToolCallDone,
)
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.shell import CommandResult, ShellState


class OrchestratorTests(unittest.TestCase):
    def test_run_command_schema_exposes_optional_output_windows(self):
        properties = RUN_COMMAND_TOOL["parameters"]["properties"]

        self.assertEqual(properties["head_chars"]["type"], "integer")
        self.assertEqual(properties["head_chars"]["minimum"], 0)
        self.assertEqual(properties["head_chars"]["maximum"], 200000)
        self.assertEqual(properties["tail_chars"]["type"], "integer")
        self.assertEqual(properties["tail_chars"]["minimum"], 0)
        self.assertEqual(properties["tail_chars"]["maximum"], 200000)
        self.assertEqual(RUN_COMMAND_TOOL["parameters"]["required"], ["command"])
        self.assertFalse(RUN_COMMAND_TOOL["parameters"]["additionalProperties"])

    def test_invalid_run_command_window_does_not_execute(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )

        with (
            patch("jarv.orchestrator.check_command", return_value=(True, "")),
            patch("jarv.orchestrator.execute_command") as execute,
        ):
            output = dispatch_tool(
                "run_command",
                {"command": "echo ok", "head_chars": -1},
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertEqual(
            output,
            "[tool argument error: head_chars must be a non-negative integer]",
        )
        execute.assert_not_called()

    def test_run_command_null_output_windows_use_defaults(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )
        command_result = CommandResult(
            "echo output",
            "a" * 40 + "MIDDLE" + "z" * 40,
            "",
            0,
        )

        with (
            patch("jarv.orchestrator.check_command", return_value=(True, "")),
            patch("jarv.orchestrator.execute_command", return_value=command_result),
        ):
            output = dispatch_tool(
                "run_command",
                {"command": "echo ok", "head_chars": None, "tail_chars": None},
                node,
                ArtifactStore(),
                client=None,
                config={**DEFAULT_CONFIG, "max_tool_output_chars": 10},
            )

        self.assertTrue(output.startswith("a" * 5))
        self.assertTrue(output.endswith("z" * 5))

    def test_run_command_rejects_oversized_output_window(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )

        with (
            patch("jarv.orchestrator.check_command", return_value=(True, "")),
            patch("jarv.orchestrator.execute_command") as execute,
        ):
            output = dispatch_tool(
                "run_command",
                {"command": "echo ok", "head_chars": 200001},
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertIn("at most 200000", output)
        execute.assert_not_called()

    def test_subagent_command_window_overrides_generic_tool_limit(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )
        calls = []
        captured_output = {}

        def fake_stream_response(*args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                yield ToolCallDone(
                    id="fc_1",
                    call_id="call_1",
                    name="run_command",
                    arguments=(
                        '{"command": "echo output", '
                        '"head_chars": 30, "tail_chars": 30}'
                    ),
                )
                yield StreamDone(response=None)
                return

            captured_output["value"] = next(
                item["output"]
                for item in args[5]
                if item.get("type") == "function_call_output"
                and item.get("call_id") == "call_1"
            )
            yield ToolCallDone(
                id="fc_2",
                call_id="call_2",
                name="finish",
                arguments='{"longform": "done", "tldr": "done"}',
            )
            yield StreamDone(response=None)

        config = {**DEFAULT_CONFIG, "max_tool_output_chars": 10}
        command_result = CommandResult(
            "echo output",
            "a" * 40 + "MIDDLE" + "z" * 40,
            "",
            0,
        )
        with (
            patch("jarv.orchestrator.stream_response", side_effect=fake_stream_response),
            patch("jarv.orchestrator.check_command", return_value=(True, "")),
            patch("jarv.orchestrator.execute_command", return_value=command_result),
        ):
            longform, tldr = run_subagent_loop(
                node,
                ArtifactStore(),
                client=None,
                config=config,
            )

        output = captured_output["value"]
        self.assertEqual((longform, tldr), ("done", "done"))
        self.assertTrue(output.startswith("a" * 30))
        self.assertTrue(output.endswith("z" * 30))
        self.assertIn("26 characters omitted from the middle", output)
        self.assertNotIn("truncated to 10 characters", output)

    def test_run_command_uses_node_shell_state(self):
        state = ShellState(cwd="C:/somewhere", env=None)
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
            shell_state=state,
        )
        command_result = CommandResult("echo ok", "ok", "", 0)

        with (
            patch("jarv.orchestrator.check_command", return_value=(True, "")),
            patch(
                "jarv.orchestrator.execute_command",
                return_value=command_result,
            ) as execute,
        ):
            dispatch_tool(
                "run_command",
                {"command": "echo ok"},
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertIs(execute.call_args.kwargs["shell_state"], state)

    def test_spawn_children_get_independent_shell_state_copies(self):
        parent = AgentNode(
            label="root",
            depth=0,
            parent_label=None,
            task="root",
            sterile=False,
            shell_state=ShellState(cwd="C:/proj", env={"A": "1"}),
        )
        seen_nodes = []

        def record_node(node, *args, **kwargs):
            seen_nodes.append(node)
            return ("long", "tldr")

        with patch("jarv.orchestrator.run_subagent_loop", side_effect=record_node):
            spawn_batch(
                parent,
                [{"label": "child", "task": "work"}],
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertEqual(len(seen_nodes), 1)
        child_state = seen_nodes[0].shell_state
        self.assertIsNot(child_state, parent.shell_state)
        self.assertIsNot(child_state.env, parent.shell_state.env)
        self.assertEqual(child_state.cwd, parent.shell_state.cwd)
        self.assertEqual(child_state.env, parent.shell_state.env)

    def test_spawn_rejects_invalid_deps_and_non_boolean_sterile(self):
        parent = AgentNode(
            label="root",
            depth=0,
            parent_label=None,
            task="root",
            sterile=False,
            visible_labels={"ready"},
        )

        with self.assertRaisesRegex(ValueError, "invalid deps"):
            spawn_batch(
                parent,
                [{"label": "child", "task": "work", "deps": ["missing"]}],
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        with self.assertRaisesRegex(ValueError, "sterile must be a boolean"):
            spawn_batch(
                parent,
                [{"label": "child", "task": "work", "sterile": "false"}],
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

    def test_spawn_rejects_too_many_children(self):
        parent = AgentNode(
            label="root",
            depth=0,
            parent_label=None,
            task="root",
            sterile=False,
        )
        children = [
            {"label": f"child_{index}", "task": "work"}
            for index in range(17)
        ]

        with self.assertRaisesRegex(ValueError, "at most 16"):
            spawn_batch(
                parent,
                children,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

    def test_subagent_read_size_overrides_generic_tool_limit(self):
        store = ArtifactStore()
        store.put("dep", "x" * 100, "dependency", "parent")
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
            visible_labels={"dep"},
        )
        calls = []
        captured_output = {}

        def fake_stream_response(*args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                yield ToolCallDone(
                    id="fc_1",
                    call_id="call_1",
                    name="read",
                    arguments='{"input": "dep", "offset": 10, "size": 20}',
                )
                yield StreamDone(response=None)
                return

            captured_output["value"] = next(
                item["output"]
                for item in args[5]
                if item.get("type") == "function_call_output"
                and item.get("call_id") == "call_1"
            )
            yield ToolCallDone(
                id="fc_2",
                call_id="call_2",
                name="finish",
                arguments='{"longform": "done", "tldr": "done"}',
            )
            yield StreamDone(response=None)

        config = {**DEFAULT_CONFIG, "max_tool_output_chars": 10}
        with patch(
            "jarv.orchestrator.stream_response",
            side_effect=fake_stream_response,
        ):
            longform, tldr = run_subagent_loop(
                node,
                store,
                client=None,
                config=config,
            )

        self.assertEqual((longform, tldr), ("done", "done"))
        self.assertIn("Requested size: 20", captured_output["value"])
        self.assertIn("Returned size: 20", captured_output["value"])
        self.assertTrue(captured_output["value"].endswith("x" * 20))
        self.assertNotIn("tool output truncated", captured_output["value"])

    def test_subagent_batches_consecutive_reads_and_preserves_call_order(self):
        store = ArtifactStore()
        store.put("first", "alpha", "first", "parent")
        store.put("second", "beta", "second", "parent")
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
            visible_labels={"first", "second"},
        )
        calls = []
        captured = {}

        def fake_stream_response(*args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                yield ToolCallDone(
                    id="fc_1",
                    call_id="call_1",
                    name="read",
                    arguments='{"input": "first", "size": 10}',
                )
                yield ToolCallDone(
                    id="fc_2",
                    call_id="call_2",
                    name="read",
                    arguments='{"input": "second", "size": 10}',
                )
                yield StreamDone(response=None)
                return

            captured["outputs"] = [
                item["output"]
                for item in args[5]
                if item.get("type") == "function_call_output"
            ]
            yield ToolCallDone(
                id="fc_3",
                call_id="call_3",
                name="finish",
                arguments='{"longform": "done", "tldr": "done"}',
            )
            yield StreamDone(response=None)

        with patch(
            "jarv.orchestrator.stream_response",
            side_effect=fake_stream_response,
        ):
            result = run_subagent_loop(
                node,
                store,
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertEqual(result, ("done", "done"))
        self.assertEqual(len(captured["outputs"]), 2)
        self.assertTrue(captured["outputs"][0].endswith("alpha"))
        self.assertTrue(captured["outputs"][1].endswith("beta"))

    def test_subagent_preserves_structured_read_output(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )
        structured_output = [
            {"type": "input_text", "text": "[READ RESULT]\nSource: local image"},
            {"type": "input_image", "image_url": "data:image/png;base64,QUJDRA=="},
        ]
        calls = []
        captured = {}

        def fake_stream_response(*args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                yield ToolCallDone(
                    id="fc_1",
                    call_id="call_1",
                    name="read",
                    arguments='{"input": "image.png"}',
                )
                yield StreamDone(response=None)
                return

            captured["output"] = next(
                item["output"]
                for item in args[5]
                if item.get("type") == "function_call_output"
            )
            yield ToolCallDone(
                id="fc_2",
                call_id="call_2",
                name="finish",
                arguments='{"longform": "done", "tldr": "done"}',
            )
            yield StreamDone(response=None)

        with (
            patch("jarv.orchestrator.stream_response", side_effect=fake_stream_response),
            patch("jarv.orchestrator.dispatch_read_tool", return_value=structured_output),
        ):
            result = run_subagent_loop(
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertEqual(result, ("done", "done"))
        self.assertEqual(captured["output"], structured_output)

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

    def test_subagent_replays_truncated_stream_once_and_discards_tool_calls(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )
        stream_count = 0

        def fake_stream_response(*_args, **_kwargs):
            nonlocal stream_count
            stream_count += 1
            if stream_count == 1:
                yield ToolCallDone(
                    id="fc_discarded",
                    call_id="call_discarded",
                    name="run_command",
                    arguments='{"command":"do not run"}',
                )
                raise RetryableStreamError("truncated")
            yield ToolCallDone(
                id="fc_finish",
                call_id="call_finish",
                name="finish",
                arguments='{"longform":"done","tldr":"done"}',
            )
            yield StreamDone(response=None)

        with (
            patch("jarv.orchestrator.stream_response", side_effect=fake_stream_response),
            patch("jarv.orchestrator.execute_command") as execute,
        ):
            result = run_subagent_loop(
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertEqual(result, ("done", "done"))
        self.assertEqual(stream_count, 2)
        execute.assert_not_called()

    def test_subagent_stops_after_one_stream_replay(self):
        node = AgentNode(
            label="child",
            depth=1,
            parent_label="root",
            task="do work",
            sterile=True,
        )
        stream_count = 0

        def fake_stream_response(*_args, **_kwargs):
            nonlocal stream_count
            stream_count += 1
            raise RetryableStreamError("still truncated")
            yield

        with patch(
            "jarv.orchestrator.stream_response",
            side_effect=fake_stream_response,
        ):
            longform, reason = run_subagent_loop(
                node,
                ArtifactStore(),
                client=None,
                config=DEFAULT_CONFIG,
            )

        self.assertIsNone(longform)
        self.assertEqual(stream_count, 2)
        self.assertEqual(reason, "provider error: still truncated")

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
