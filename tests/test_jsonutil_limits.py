"""JSON recovery rejects payloads the decoder cannot inspect safely."""

import json
import sys

import pytest

from jarv.jsonutil import iter_json_objects, salvage_json_object


@pytest.fixture(params=["nesting", "integer"])
def excessive_json(request):
    if request.param == "integer":
        get_limit = getattr(sys, "get_int_max_str_digits", lambda: 0)
        limit = get_limit()
        if not limit:
            pytest.skip("interpreter does not limit integer string conversion")
        return '{"value":' + "9" * (limit + 1) + "}"
    depth = sys.getrecursionlimit() + 100
    payload = '{"value":' + "[" * depth + "0" + "]" * depth + "}"
    try:
        json.loads(payload)
    except RecursionError:
        return payload
    pytest.skip("interpreter's JSON decoder does not use the Python recursion limit")


@pytest.mark.parametrize("prefix,suffix", [
    ("", ""), ("```json\n", "\n```"), ("Here: ", ""),
    ('{"ok":true} trailing ', ""),
])
def test_salvage_rejects_decoder_limits(excessive_json, prefix, suffix):
    assert salvage_json_object(prefix + excessive_json + suffix) is None


def test_verdict_scan_stops_on_decoder_limits(excessive_json):
    assert list(iter_json_objects(excessive_json + ' {"allow":true}')) == []
