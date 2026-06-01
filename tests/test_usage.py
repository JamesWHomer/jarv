import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from jarv.usage import (
    append_global_usage_record,
    aggregate_usage_records,
    load_global_usage_records,
    load_usage,
    record_response_usage,
)


class UsageRecordingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
