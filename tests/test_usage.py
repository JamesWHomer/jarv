import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from jarv import model_catalog
from jarv.model_catalog import CatalogModel
from jarv.usage import (
    append_global_usage_record,
    aggregate_usage_records,
    estimate_token_cost_usd,
    format_tokens_compact,
    known_context_window,
    load_global_usage_records,
    load_usage,
    record_response_usage,
    usage_cost_summary,
)


class UsageRecordingTests(unittest.TestCase):
    def setUp(self):
        self.catalog_dir = TemporaryDirectory()
        self.cache_patch = patch.object(
            model_catalog,
            "CACHE_DIR",
            Path(self.catalog_dir.name),
        )
        self.cache_patch.start()
        model_catalog._write_cache("openrouter", [
            CatalogModel(
                id="openai/gpt-5.4-mini",
                metadata={
                    "context_length": 272_000,
                    "pricing": {
                        "prompt": "0.00000075",
                        "input_cache_read": "0.000000075",
                        "completion": "0.0000045",
                    },
                },
            ),
            CatalogModel(
                id="anthropic/claude-sonnet-4.6",
                metadata={
                    "context_length": 1_000_000,
                    "pricing": {
                        "prompt": "0.000003",
                        "input_cache_read": "0.0000003",
                        "completion": "0.000015",
                    },
                },
            ),
            CatalogModel(
                id="google/gemini-3-flash-preview",
                metadata={
                    "context_length": 1_048_576,
                    "pricing": {
                        "prompt": "0.0000005",
                        "input_cache_read": "0.00000005",
                        "completion": "0.000003",
                    },
                },
            ),
            CatalogModel(
                id="openrouter/free",
                metadata={
                    "context_length": 128_000,
                    "pricing": {
                        "prompt": "0",
                        "input_cache_read": "0",
                        "completion": "0",
                    },
                },
            ),
        ])

    def tearDown(self):
        self.cache_patch.stop()
        self.catalog_dir.cleanup()

    def test_openrouter_metadata_prices_cached_input_separately(self):
        record = {
            "input_tokens": 1_000_000,
            "cached_input_tokens": 500_000,
            "output_tokens": 100_000,
        }
        self.assertEqual(known_context_window("gpt-5.4-mini"), 272_000)
        self.assertAlmostEqual(
            estimate_token_cost_usd(record, "gpt-5.4-mini"),
            0.8625,
        )

    def test_records_estimated_usage_when_provider_omits_usage(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"

            record_response_usage(
                usage_path,
                "session-id",
                "unknown-provider/model",
                response=None,
                source="root",
                context_breakdown={
                    "system": 10,
                    "tools": 20,
                    "history": 30,
                    "tool_io": 0,
                    "reasoning": 0,
                },
                output_text="hello there",
                record_global=False,
            )

            usage = load_usage(usage_path, "session-id")

        totals = usage["totals"]
        self.assertEqual(totals["request_count"], 1)
        self.assertEqual(totals["input_tokens"], 60)
        self.assertGreater(totals["output_tokens"], 0)
        self.assertTrue(usage["last_request"]["estimated"])

    def test_exact_provider_usage_wins_over_estimate(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            response = SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=3,
                    total_tokens=15,
                )
            )

            record_response_usage(
                usage_path,
                "session-id",
                "test-model",
                response=response,
                source="root",
                context_breakdown={
                    "system": 100,
                    "tools": 100,
                    "history": 100,
                    "tool_io": 100,
                    "reasoning": 100,
                },
                output_text="this estimate should not be used",
                record_global=False,
            )

            usage = load_usage(usage_path, "session-id")

        self.assertEqual(usage["totals"]["input_tokens"], 12)
        self.assertEqual(usage["totals"]["output_tokens"], 3)
        self.assertNotIn("estimated", usage["last_request"])

    def test_records_served_tier_without_using_standard_price_estimates(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            response = SimpleNamespace(
                service_tier="priority",
                usage=SimpleNamespace(
                    input_tokens=12,
                    output_tokens=3,
                    total_tokens=15,
                ),
            )

            record_response_usage(
                usage_path,
                "session-id",
                "gpt-5.4-mini",
                response=response,
                source="root",
                provider="openai",
                requested_service_tier="priority",
                record_global=False,
            )

            usage = load_usage(usage_path, "session-id")

        self.assertEqual(usage["last_request"]["requested_service_tier"], "priority")
        self.assertEqual(usage["last_request"]["served_service_tier"], "priority")
        self.assertEqual(usage["last_request"]["cost_status"], "estimated")

    def test_openai_tier_prices_are_applied_per_request(self):
        record = {
            "provider": "openai",
            "input_tokens": 1_000_000,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        }
        self.assertAlmostEqual(
            estimate_token_cost_usd(
                {**record, "served_service_tier": "standard"},
                "gpt-5.4-mini",
            ),
            5.25,
        )
        self.assertAlmostEqual(
            estimate_token_cost_usd(
                {**record, "served_service_tier": "flex"},
                "gpt-5.4-mini",
            ),
            2.625,
        )
        self.assertAlmostEqual(
            estimate_token_cost_usd(
                {**record, "served_service_tier": "priority"},
                "gpt-5.4-mini",
            ),
            10.5,
        )

    def test_mixed_tiers_sum_stored_request_costs(self):
        records = [
            {
                "model": "gpt-5.4-mini",
                "provider": "openai",
                "source": "root",
                "served_service_tier": "flex",
                "input_tokens": 1_000_000,
                "cached_input_tokens": 0,
                "uncached_input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 2_000_000,
                "estimated_cost_usd": 2.625,
                "cost_status": "estimated",
            },
            {
                "model": "gpt-5.4-mini",
                "provider": "openai",
                "source": "root",
                "served_service_tier": "priority",
                "input_tokens": 1_000_000,
                "cached_input_tokens": 0,
                "uncached_input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 2_000_000,
                "estimated_cost_usd": 10.5,
                "cost_status": "estimated",
            },
        ]

        aggregate = aggregate_usage_records(records)
        cost = usage_cost_summary(aggregate["totals"])

        self.assertAlmostEqual(cost["total_usd"], 13.125)
        self.assertEqual(cost["estimated_requests"], 2)
        self.assertAlmostEqual(
            usage_cost_summary(aggregate["tiers"]["flex"])["total_usd"],
            2.625,
        )
        self.assertAlmostEqual(
            usage_cost_summary(aggregate["tiers"]["priority"])["total_usd"],
            10.5,
        )

    def test_openrouter_provider_reported_cost_wins(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            response = {
                "service_tier": "priority",
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                    "cost": 7.125,
                },
            }

            record_response_usage(
                usage_path,
                "session-id",
                "vendor/model",
                response=response,
                source="root",
                provider="openrouter",
                requested_service_tier="priority",
                record_global=False,
            )
            usage = load_usage(usage_path, "session-id")

        self.assertEqual(usage["last_request"]["cost_status"], "exact")
        self.assertEqual(usage["last_request"]["provider_cost_usd"], 7.125)
        self.assertEqual(usage["totals"]["cost_exact_request_count"], 1)

    def test_openrouter_zero_cost_for_priced_model_is_estimated(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            response = {
                "model": "openai/gpt-5.4-mini",
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                    "cost": 0,
                },
            }

            record_response_usage(
                usage_path,
                "session-id",
                "openai/gpt-5.4-mini",
                response=response,
                source="root",
                provider="openrouter",
                record_global=False,
            )
            usage = load_usage(usage_path, "session-id")

        self.assertEqual(usage["last_request"]["cost_status"], "estimated")
        self.assertEqual(usage["last_request"]["provider_reported_cost_usd"], 0)
        self.assertNotIn("provider_cost_usd", usage["last_request"])
        self.assertAlmostEqual(usage["last_request"]["estimated_cost_usd"], 5.25)
        self.assertEqual(usage["totals"]["cost_estimated_request_count"], 1)

    def test_openrouter_zero_cost_for_free_model_stays_exact(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            response = {
                "model": "openrouter/free",
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                    "cost": 0,
                },
            }

            record_response_usage(
                usage_path,
                "session-id",
                "openrouter/free",
                response=response,
                source="root",
                provider="openrouter",
                record_global=False,
            )
            usage = load_usage(usage_path, "session-id")

        self.assertEqual(usage["last_request"]["cost_status"], "exact")
        self.assertEqual(usage["last_request"]["provider_cost_usd"], 0)
        self.assertEqual(usage["last_request"]["provider_reported_cost_usd"], 0)
        self.assertNotIn("estimated_cost_usd", usage["last_request"])
        self.assertEqual(usage["totals"]["cost_exact_request_count"], 1)

    def test_openrouter_zero_cost_without_catalog_price_is_unknown(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            response = {
                "model": "unknown/vendor-model",
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                    "cost": 0,
                },
            }

            record_response_usage(
                usage_path,
                "session-id",
                "unknown/vendor-model",
                response=response,
                source="root",
                provider="openrouter",
                record_global=False,
            )
            usage = load_usage(usage_path, "session-id")

        self.assertEqual(usage["last_request"]["cost_status"], "unknown")
        self.assertEqual(usage["last_request"]["provider_reported_cost_usd"], 0)
        self.assertNotIn("provider_cost_usd", usage["last_request"])
        self.assertNotIn("estimated_cost_usd", usage["last_request"])
        self.assertEqual(usage["totals"]["cost_unknown_request_count"], 1)

    def test_openrouter_router_records_served_model(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            response = {
                "model": "anthropic/claude-sonnet-4.6",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                    "cost": 0.001,
                },
            }

            record_response_usage(
                usage_path,
                "session-id",
                "openrouter/auto",
                response=response,
                source="root",
                provider="openrouter",
                record_global=False,
            )
            usage = load_usage(usage_path, "session-id")

        self.assertEqual(
            usage["last_request"]["model"],
            "anthropic/claude-sonnet-4.6",
        )
        self.assertEqual(
            usage["last_request"]["requested_model"],
            "openrouter/auto",
        )
        self.assertIn("anthropic/claude-sonnet-4.6", usage["models"])
        self.assertNotIn("openrouter/auto", usage["models"])

    def test_unreported_priority_downgrade_is_not_guessed(self):
        record = {
            "provider": "openai",
            "requested_service_tier": "priority",
            "input_tokens": 1_000_000,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        }
        self.assertIsNone(estimate_token_cost_usd(record, "gpt-5.4-mini"))
        self.assertAlmostEqual(
            estimate_token_cost_usd(
                {**record, "served_service_tier": "standard"},
                "gpt-5.4-mini",
            ),
            5.25,
        )

    def test_anthropic_priority_is_contract_priced(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            response = {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "service_tier": "priority",
                },
            }
            record_response_usage(
                usage_path,
                "session-id",
                "claude-sonnet-4-6",
                response=response,
                source="root",
                provider="anthropic",
                requested_service_tier="priority",
                record_global=False,
            )
            usage = load_usage(usage_path, "session-id")

        self.assertEqual(usage["last_request"]["cost_status"], "contract")
        self.assertEqual(usage["totals"]["cost_contract_request_count"], 1)

    def test_gemini_flex_keeps_cached_input_at_standard_rate(self):
        cost = estimate_token_cost_usd(
            {
                "provider": "gemini",
                "served_service_tier": "flex",
                "input_tokens": 1_000_000,
                "cached_input_tokens": 500_000,
                "uncached_input_tokens": 500_000,
                "output_tokens": 1_000_000,
                "reasoning_output_tokens": 100_000,
            },
            "gemini-3-flash-preview",
        )
        self.assertAlmostEqual(cost, 1.8)

    def test_global_usage_records_are_appended_alongside_session_usage(self):
        with TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "usage-test.json"
            global_path = Path(tmp) / "usage.json"
            response = SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=3,
                    total_tokens=15,
                )
            )

            record_response_usage(
                usage_path,
                "session-id",
                "test-model",
                response=response,
                source="root",
                global_usage_path=global_path,
            )
            record_response_usage(
                usage_path,
                "session-id",
                "test-model",
                response=response,
                source="subagent",
                global_usage_path=global_path,
            )

            session_usage = load_usage(usage_path, "session-id")
            global_records = load_global_usage_records(global_path)
            aggregate = aggregate_usage_records(global_records)
            legacy_json_exists = global_path.exists()
            jsonl_exists = global_path.with_suffix(".jsonl").exists()

        self.assertEqual(session_usage["totals"]["request_count"], 2)
        self.assertFalse(legacy_json_exists)
        self.assertTrue(jsonl_exists)
        self.assertEqual(len(global_records), 2)
        self.assertEqual([record["source"] for record in global_records], ["root", "subagent"])
        self.assertEqual(aggregate["totals"]["total_tokens"], 30)
        self.assertEqual(aggregate["sources"]["root"]["request_count"], 1)
        self.assertEqual(aggregate["sources"]["subagent"]["request_count"], 1)

    def test_global_usage_records_combine_legacy_json_and_jsonl(self):
        with TemporaryDirectory() as tmp:
            global_path = Path(tmp) / "usage.json"
            global_path.write_text(
                '{"version":1,"records":[{"created_at":"2026-05-28T00:00:00Z","session_id":"legacy","model":"test-model","source":"root","input_tokens":1,"cached_input_tokens":0,"uncached_input_tokens":1,"output_tokens":2,"reasoning_output_tokens":0,"total_tokens":3}]}',
                encoding="utf-8",
            )
            append_global_usage_record(
                {
                    "created_at": "2026-05-29T00:00:00Z",
                    "session_id": "jsonl",
                    "model": "test-model",
                    "source": "root",
                    "input_tokens": 4,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 4,
                    "output_tokens": 5,
                    "reasoning_output_tokens": 0,
                    "total_tokens": 9,
                },
                global_path,
            )

            records = load_global_usage_records(global_path)

        self.assertEqual([record["session_id"] for record in records], ["legacy", "jsonl"])

    def test_global_usage_records_can_be_filtered_by_time_window(self):
        with TemporaryDirectory() as tmp:
            global_path = Path(tmp) / "usage.json"
            append_global_usage_record(
                {
                    "created_at": "2026-05-28T00:00:00Z",
                    "session_id": "old",
                    "model": "test-model",
                    "source": "root",
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 10,
                    "output_tokens": 2,
                    "reasoning_output_tokens": 0,
                    "total_tokens": 12,
                },
                global_path,
            )
            append_global_usage_record(
                {
                    "created_at": "2026-05-29T01:00:00Z",
                    "session_id": "new",
                    "model": "test-model",
                    "source": "root",
                    "input_tokens": 20,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 20,
                    "output_tokens": 4,
                    "reasoning_output_tokens": 0,
                    "total_tokens": 24,
                },
                global_path,
            )

            records = load_global_usage_records(
                global_path,
                since=timedelta(hours=24),
                now=datetime(2026, 5, 29, 2, 0, tzinfo=timezone.utc),
            )

        self.assertEqual([record["session_id"] for record in records], ["new"])

    def test_malformed_or_missing_global_usage_fails_soft(self):
        with TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.json"
            malformed_path = Path(tmp) / "usage.json"
            malformed_path.write_text("{not json", encoding="utf-8")

            self.assertEqual(load_global_usage_records(missing_path, warn=False), [])
            self.assertEqual(load_global_usage_records(malformed_path, warn=False), [])


class FormatTokensCompactTests(unittest.TestCase):
    def test_compact_token_formatting(self):
        self.assertEqual(format_tokens_compact(0), "0")
        self.assertEqual(format_tokens_compact(None), "0")
        self.assertEqual(format_tokens_compact(5_200), "5,200")
        self.assertEqual(format_tokens_compact(9_999), "9,999")
        self.assertEqual(format_tokens_compact(12_300), "12.3K")
        self.assertEqual(format_tokens_compact(340_000), "340K")
        self.assertEqual(format_tokens_compact(1_240_000), "1.24M")
        self.assertEqual(format_tokens_compact(2_000_000), "2M")


if __name__ == "__main__":
    unittest.main()
