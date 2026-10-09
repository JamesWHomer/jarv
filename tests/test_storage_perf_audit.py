"""Behavioral coverage for storage copies and usage aggregation shortcuts."""

import copy
import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from jarv import storage, usage, usage_view


@pytest.mark.parametrize("with_snapshot", [False, True])
def test_staged_payload_is_isolated_from_input_and_read_snapshots(tmp_path, with_snapshot):
    path = tmp_path / "records.json"
    value = storage.JsonDict({"nested": {"items": ["saved"]}})
    value.baseline = None
    with storage.transaction(path):
        storage.write_json(path, value, snapshot=value if with_snapshot else None)
        value["nested"]["items"].append("not saved")
        loaded = storage.read_json(path, {}, dict)
        loaded["nested"]["items"].append("read changed")
        loaded.baseline["nested"]["items"].append("baseline changed")
        assert storage.read_json(path, {}, dict) == {"nested": {"items": ["saved"]}}

    assert json.loads(path.read_text(encoding="utf-8")) == {"nested": {"items": ["saved"]}}
    if with_snapshot:
        assert value.baseline == {"nested": {"items": ["saved"]}}
        # A subsequent save still compares with the value actually committed.
        storage.write_json(path, value, snapshot=value)
        assert storage.read_json(path, {}, dict)["nested"]["items"] == ["saved", "not saved"]


def test_unsaved_read_changes_do_not_change_transaction_baseline(tmp_path):
    path = tmp_path / "records.json"
    path.write_text('{"nested":{"items":["saved"]}}', encoding="utf-8")
    with storage.transaction(path):
        loaded = storage.read_json(path, {}, dict)
        loaded["nested"]["items"].append("not saved")
        loaded.baseline["nested"]["items"].append("baseline changed")
        storage.write_json(path, {"nested": {"items": ["replacement"]}})
    assert storage.read_json(path, {}, dict) == {"nested": {"items": ["replacement"]}}


def test_usage_totals_match_full_aggregation_and_record_normalization():
    rng = random.Random(218)
    records = [None, "invalid", {}]
    for index in range(100):
        record = {
            "input_tokens": rng.randrange(1000),
            "output_tokens": rng.randrange(1000),
            "source": rng.choice(["root", "child", None]),
            "provider": rng.choice(["openai", "anthropic", None]),
            "model": f"model-{index % 7}",
        }
        if index % 2:
            record["cached_input_tokens"] = rng.randrange(record["input_tokens"] + 1)
        if index % 3:
            record["provider_cost_usd"] = rng.random()
        if index % 4:
            record["estimated_cost_usd"] = rng.random()
        if index % 5:
            record["cost_status"] = rng.choice(["exact", "estimated", "unknown", "contract"])
        records.append(record)

    expected_records = copy.deepcopy(records)
    expected = usage.aggregate_usage_records(expected_records)["totals"]
    assert usage.aggregate_usage_totals(records) == expected
    assert records == expected_records
    assert usage.aggregate_usage_totals([]) == usage.aggregate_usage_records([])["totals"]


@pytest.mark.parametrize("window", [None, timedelta(hours=1), timedelta(days=7)])
def test_usage_window_split_preserves_boundaries_order_and_invalid_records(window, monkeypatch):
    now = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    records = [None, {}, {"created_at": "invalid"}]
    for hours in (200, 168, 2, 1, 0, -1, 336):
        instant = now - timedelta(hours=hours)
        records.append({"created_at": instant.isoformat(), "id": hours})
    records.extend([
        {"created_at": "2026-10-10T11:00:00"},
        {"created_at": "2026-10-10T21:00:00+10:00"},
    ])
    expected = (
        (list(records), None) if window is None else (
            usage_view._filter_records(records, window, now),
            usage_view._filter_records(records, window * 2, now, until=now - window),
        )
    )
    parsed = 0
    original = usage_view.parse_timestamp

    def count(value):
        nonlocal parsed
        parsed += 1
        return original(value)

    monkeypatch.setattr(usage_view, "parse_timestamp", count)
    assert usage_view._split_windows(records, window, now) == expected
    assert parsed == (0 if window is None else sum(isinstance(record, dict) for record in records))
