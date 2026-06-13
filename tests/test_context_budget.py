import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarv import model_catalog
from jarv.agent import to_response_input_item
from jarv.config import DEFAULT_CONFIG
from jarv.context_budget import (
    build_input,
    compact_history_view,
    compact_oldest_turns,
    context_budget_status,
    estimate_history_tokens,
    summarize_turn_items,
    trim_items_to_budget,
    trim_turn_input,
)
from jarv.model_catalog import CatalogModel
from jarv.usage import estimate_item_tokens, resolve_context_window


class ContextBudgetTests(unittest.TestCase):
    def _tiny_config(self) -> dict:
        return {
            **DEFAULT_CONFIG,
            "context_window_fallback": 400,
            "context_budget_ratio": 0.75,
            "context_output_reserve_ratio": 0.15,
            "context_compaction_threshold": 0.85,
        }

    def test_trim_items_keeps_recent_suffix_within_budget(self):
        model = "gpt-5.4-mini"
        items = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        budget = estimate_item_tokens(model, items[-2]) + estimate_item_tokens(model, items[-1])

        trimmed = trim_items_to_budget(items, model, budget)

        self.assertEqual(
            trimmed,
            [
                {"role": "user", "content": "recent question"},
                {"role": "assistant", "content": "recent answer"},
            ],
        )

    def test_trim_items_aligns_to_user_boundary(self):
        model = "gpt-5.4-mini"
        items = [
            {"type": "function_call_output", "call_id": "1", "output": "orphan"},
            {"role": "user", "content": "hello"},
        ]
        trimmed = trim_items_to_budget(items, model, 10_000)
        self.assertEqual(trimmed, [{"role": "user", "content": "hello"}])

    def test_build_input_uses_token_budget_not_item_count(self):
        config = self._tiny_config()
        model = "unknown-model"
        instructions = "system prompt"
        tools = [{"type": "function", "name": "run_command"}]
        history = [
            {"role": "user", "content": "old " + ("x" * 600)},
            {"role": "assistant", "content": "old answer " + ("y" * 600)},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "newer"},
        ]

        api_items = build_input(
            history,
            model=model,
            config=config,
            instructions=instructions,
            tools=tools,
        )

        self.assertEqual(
            api_items,
            [
                {"role": "user", "content": "new"},
                {"role": "assistant", "content": "newer"},
            ],
        )
        self.assertEqual(len(history), 4)

    def test_build_input_drops_orphaned_tool_pairs_when_budget_is_tight(self):
        config = {**self._tiny_config(), "context_window_fallback": 80}
        model = "unknown-model"
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
                "output": "x" * 2000,
            },
            {"role": "assistant", "content": "done"},
        ]

        api_items = build_input(
            history,
            model=model,
            config=config,
            instructions="",
            tools=[],
        )

        self.assertEqual(api_items, [])

    def test_trim_turn_input_limits_within_turn_growth(self):
        config = self._tiny_config()
        model = "unknown-model"
        instructions = "system"
        tools = []
        input_items = [
            {"role": "user", "content": "start " + ("z" * 1200)},
            {
                "type": "function_call",
                "id": "call_1",
                "call_id": "call_1",
                "name": "run_command",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "output " + ("o" * 1200),
            },
        ]

        trimmed = trim_turn_input(
            input_items,
            model=model,
            config=config,
            instructions=instructions,
            tools=tools,
        )

        self.assertLess(len(trimmed), len(input_items))
        self.assertEqual(trimmed[0]["role"], "user")

    def test_summarize_turn_items_includes_roles(self):
        summary = summarize_turn_items([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ])
        self.assertIn("User: hello", summary)
        self.assertIn("Assistant: world", summary)

    def test_compact_oldest_turns_replaces_prefix_with_summary(self):
        config = self._tiny_config()
        model = "unknown-model"
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "second answer"},
        ]
        original = [dict(item) for item in history]

        modified = compact_oldest_turns(
            history,
            model=model,
            config=config,
            instructions="",
            tools=[],
            target_tokens=1,
        )

        self.assertTrue(modified)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["type"], "compacted_summary")
        self.assertEqual(to_response_input_item(history[0]), {
            "role": "user",
            "content": history[0]["content"],
        })

    def test_compact_history_view_does_not_mutate_stored_history(self):
        config = self._tiny_config()
        model = "unknown-model"
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "second answer"},
        ]
        original = [dict(item) for item in history]

        view = compact_history_view(
            history,
            model=model,
            config=config,
            instructions="",
            tools=[],
            target_tokens=1,
        )

        self.assertEqual(history, original)
        self.assertEqual(len(view), 3)
        self.assertEqual(view[0]["type"], "compacted_summary")

    def test_build_input_compacts_without_mutating_history(self):
        config = {
            **self._tiny_config(),
            "context_compaction_threshold": 0.5,
        }
        model = "unknown-model"
        history = [
            {"role": "user", "content": "one " + ("a" * 500)},
            {"role": "assistant", "content": "two " + ("b" * 500)},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        original = [dict(item) for item in history]

        api_items = build_input(
            history,
            model=model,
            config=config,
            instructions="instructions " + ("i" * 200),
            tools=[{"type": "function", "name": "run_command"}],
        )

        self.assertEqual(history, original)
        self.assertTrue(api_items[0]["content"].startswith("[Compacted earlier conversation]"))
        self.assertEqual(api_items[-2]["content"], "three")
        self.assertEqual(api_items[-1]["content"], "four")

    def test_resolve_context_window_prefers_metadata_then_fallback(self):
        with TemporaryDirectory() as tmp, patch.object(
            model_catalog,
            "CACHE_DIR",
            Path(tmp),
        ):
            model_catalog._write_cache("openrouter", [
                CatalogModel(
                    id="openai/gpt-5.4-mini",
                    metadata={"context_length": 272_000},
                ),
            ])
            self.assertEqual(resolve_context_window("gpt-5.4-mini", {}), 272_000)
        self.assertEqual(
            resolve_context_window("totally-unknown", {"context_window_fallback": 999}),
            999,
        )

    def test_context_budget_status_reports_fill_ratio(self):
        config = self._tiny_config()
        status = context_budget_status(
            "unknown-model",
            config,
            instructions="x" * 80,
            tools=[],
            history=[{"role": "user", "content": "y" * 80}],
        )
        self.assertEqual(status["context_window"], 400)
        self.assertGreater(status["history_tokens"], 0)
        self.assertIsNotNone(status["fill_ratio"])


if __name__ == "__main__":
    unittest.main()
