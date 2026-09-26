import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarv import model_catalog, models_dev
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
from jarv.retained_outputs import RetainedOutputStore
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

    def test_build_input_preserves_tool_pairs_when_budget_is_tight(self):
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

        self.assertEqual([i.get("type") for i in api_items],
                         [None, "function_call", "function_call_output", None])
        self.assertEqual(api_items[1]["call_id"], api_items[2]["call_id"])

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

        store = RetainedOutputStore()
        trimmed = trim_turn_input(
            input_items,
            model=model,
            config=config,
            instructions=instructions,
            tools=tools,
            retained_store=store,
        )

        self.assertEqual(len(trimmed), len(input_items))
        self.assertEqual(trimmed[0]["role"], "user")
        self.assertEqual(trimmed[1], input_items[1])
        self.assertEqual(trimmed[2]["call_id"], "call_1")
        ref = trimmed[2]["output"].split(": ", 1)[1].split("]", 1)[0]
        self.assertEqual(store.get(ref).content, input_items[2]["output"])
        self.assertEqual(trim_turn_input(
            trimmed, model=model, config=config, instructions=instructions,
            tools=tools, retained_store=store,
        ), trimmed)

    def test_compaction_advances_past_summaries_and_preserves_latest_turn(self):
        history = []
        for label in ("first", "second", "latest"):
            history.extend([
                {"role": "user", "content": label},
                {"role": "assistant", "content": label + " x" * 1000},
            ])
        latest = history[-2:]
        kwargs = dict(model="unknown-model", config=self._tiny_config(),
                      instructions="", tools=[], target_tokens=1)
        compact_oldest_turns(history, **kwargs)
        self.assertEqual(len(history), 4)
        self.assertIn("User: first", history[0]["content"])
        self.assertIn("User: second", history[1]["content"])
        self.assertEqual(history[-2:], latest)
        self.assertFalse(compact_oldest_turns(history, **kwargs))
        self.assertEqual(history[0]["content"].count("[Compacted earlier conversation]"), 1)

    def test_active_parallel_pairs_survive_repeated_trimming(self):
        store = RetainedOutputStore()
        items = [{"role": "user", "content": "perform the edits"}]
        for number in range(3):
            calls = [
                {"type": "function_call", "call_id": f"{number}-{i}",
                 "name": "edit", "arguments": '{"path":"file.txt"}'}
                for i in range(2)
            ]
            outputs = [
                {"type": "function_call_output", "call_id": call["call_id"],
                 "output": "Edit succeeded\n" + "details " * 2000}
                for call in calls
            ]
            items = trim_turn_input(
                items + calls + outputs, model="unknown-model",
                config=self._tiny_config(), instructions="", tools=[],
                retained_store=store,
            )
            self.assertEqual(len(items), 1 + (number + 1) * 4)
            self.assertEqual(
                [i["call_id"] for i in items if i.get("type") == "function_call"],
                [i["call_id"] for i in items if i.get("type") == "function_call_output"],
            )
        self.assertEqual(trim_turn_input(
            items, model="unknown-model", config=self._tiny_config(),
            instructions="", tools=[],
        ), items)

    def test_summarize_turn_items_includes_roles(self):
        summary = summarize_turn_items([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ])
        self.assertIn("User: hello", summary)
        self.assertIn("Assistant: world", summary)

    def test_structured_image_outputs_are_summarized_for_budgeting(self):
        image_output = [
            {"type": "input_text", "text": "[READ RESULT]\nSource: local image"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64," + ("A" * 4000),
            },
        ]

        summary = summarize_turn_items([
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": image_output,
            }
        ])
        tokens = estimate_item_tokens(
            "unknown-model",
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": image_output,
            },
        )

        self.assertIn("[image output 1: image/png", summary)
        self.assertNotIn("A" * 100, summary)
        self.assertLess(tokens, 100)

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

    def test_resolve_context_window_prefers_provider_then_catalog_then_fallback(self):
        config = {"provider": "openai"}
        with TemporaryDirectory() as tmp, patch.object(
            model_catalog,
            "CACHE_DIR",
            Path(tmp),
        ):
            # What the provider itself advertised outranks the catalog.
            model_catalog._write_cache("openai", [
                CatalogModel(
                    id="gpt-5.4-mini",
                    metadata={"context_length": 272_000},
                ),
            ])
            self.assertEqual(resolve_context_window("gpt-5.4-mini", config), 272_000)

            # With the provider silent, models.dev answers.
            model_catalog._write_cache("openai", [])
            self.assertEqual(
                resolve_context_window("gpt-5.4-mini", config),
                models_dev.lookup("openai", "gpt-5.4-mini").context_limit,
            )

        self.assertEqual(
            resolve_context_window("totally-unknown", {"context_window_fallback": 999}),
            999,
        )

    def test_context_budget_uses_endpoint_scoped_catalog_window(self):
        model = "endpoint-test-model"
        default_config = {"provider": "openai"}
        small_config = {**default_config, "base_url": "https://small.example/v1"}
        large_config = {**default_config, "base_url": "https://large.example/v1"}
        cases = [
            (default_config, 131_072),
            (small_config, 4_096),
            (large_config, 262_144),
        ]
        with TemporaryDirectory() as tmp, patch.object(
            model_catalog, "CACHE_DIR", Path(tmp),
        ):
            for config, window in cases:
                model_catalog._write_cache(config, [
                    CatalogModel(id=model, metadata={"context_length": window}),
                ])

            for config, window in cases:
                with self.subTest(config=config):
                    config = {
                        **config,
                        "context_budget_ratio": 0.75,
                        "context_output_reserve_ratio": 0.25,
                    }
                    self.assertEqual(resolve_context_window(model, config), window)
                    status = context_budget_status(
                        model, config, instructions="", tools=[], history=[],
                    )
                    self.assertEqual(status["context_window"], window)
                    self.assertEqual(
                        status["history_budget"], window // 2 - status["fixed_tokens"],
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
