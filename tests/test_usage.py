import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from jarv.usage import load_usage, record_response_usage


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
            )

            usage = load_usage(usage_path, "session-id")

        self.assertEqual(usage["totals"]["input_tokens"], 12)
        self.assertEqual(usage["totals"]["output_tokens"], 3)
        self.assertNotIn("estimated", usage["last_request"])


if __name__ == "__main__":
    unittest.main()
