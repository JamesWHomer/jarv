"""Use documented server inference rates, including short responses."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jarv.generation_speed import provider_output_speed
from jarv.provider import _stream_chat_completions
from jarv.usage import usage_from_response


@pytest.mark.parametrize("response,expected", [
    ({"usage": {"completion_tokens": 2, "completion_time": 0.01}}, 200),
    ({"x_groq": {"usage": {"completion_tokens": 20, "completion_time": 0.1}}}, 200),
    ({"timings": {"predicted_per_second": 123.5}}, 123.5),
    ({"timings": {"predicted_n": 3, "predicted_ms": 15}}, 200),
    ({"eval_count": 5, "eval_duration": 25_000_000}, 200),
    ({"stats": {"tokens_per_second": 51.4, "generation_time": 0.95}}, 51.4),
])
def test_documented_server_rate_metrics(response, expected):
    assert provider_output_speed(response) == pytest.approx(expected)


def test_server_reported_rate_wins_over_recalculation():
    response = {"timings": {
        "predicted_per_second": 150, "predicted_n": 2, "predicted_ms": 10,
    }}
    assert provider_output_speed(response) == 150


def test_metrics_accept_sdk_objects_and_numeric_strings():
    response = SimpleNamespace(usage=SimpleNamespace(completion_tokens="12", completion_time="0.06"))
    assert provider_output_speed(response) == 200


@pytest.mark.parametrize("invalid", [None, True, False, -1, "bad", float("inf"), float("nan"), 10 ** 400])
@pytest.mark.parametrize("container,key", [("timings", "predicted_per_second"), ("stats", "tokens_per_second")])
def test_invalid_direct_rate_is_unavailable(container, key, invalid):
    assert provider_output_speed({container: {key: invalid}}) is None


@pytest.mark.parametrize("invalid", [0, -1, True, float("inf"), float("nan"), "bad"])
@pytest.mark.parametrize("response_builder", [
    lambda value: {"usage": {"completion_tokens": 2, "completion_time": value}},
    lambda value: {"timings": {"predicted_n": 2, "predicted_ms": value}},
    lambda value: {"eval_count": 2, "eval_duration": value},
    lambda value: {"timings": {"predicted_per_second": 0, "predicted_ms": value}},
    lambda value: {"stats": {"tokens_per_second": 0, "generation_time": value}},
])
def test_invalid_generation_duration_is_unavailable(response_builder, invalid):
    assert provider_output_speed(response_builder(invalid)) is None


@pytest.mark.parametrize("count", [True, -1, 1.5, float("nan"), float("inf"), "bad"])
def test_invalid_or_fractional_token_count_is_unavailable(count):
    assert provider_output_speed({"usage": {"completion_tokens": count, "completion_time": 1}}) is None


@pytest.mark.parametrize("response", [
    {"usage": {"completion_tokens": 0, "completion_time": 0.1}},
    {"timings": {"predicted_n": 0, "predicted_ms": 100}},
    {"eval_count": 0, "eval_duration": 100_000_000},
    {"stats": {"tokens_per_second": 0}},
])
def test_measured_zero_output_rate_is_valid(response):
    assert provider_output_speed(response) == 0


def test_missing_metrics_do_not_use_total_request_time():
    assert provider_output_speed({"usage": {"completion_tokens": 3, "total_time": 2}}) is None


def test_caller_owned_metadata_is_not_trusted_as_server_timing():
    assert provider_output_speed({
        "usage": {"output_tokens": 4}, "metadata": {"completion_time": "0.02"},
    }) is None


def test_nonfinite_computed_rate_is_unavailable():
    assert provider_output_speed({"usage": {"completion_tokens": 10, "completion_time": 5e-324}}) is None


def stream_response(chunks):
    with patch("jarv.openai_http.stream_chat", return_value=iter(chunks)):
        return list(_stream_chat_completions(object(), "model", "system", [], []))[-1].response


def test_groq_stream_normalizes_nested_usage_and_keeps_metadata():
    response = stream_response([
        {"x_groq": {"id": "groq-request"}, "choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]},
        {"choices": [], "x_groq": {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "completion_time": 0.01}}},
    ])

    assert response["x_groq"]["id"] == "groq-request"
    assert usage_from_response(response)["total_tokens"] == 12
    assert provider_output_speed(response) == 200


def test_groq_nested_usage_does_not_overwrite_canonical_usage():
    response = stream_response([{
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "cost": 0.1},
        "x_groq": {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "completion_time": 0.02}},
    }])

    assert response["usage"]["prompt_tokens"] == 20
    assert response["usage"]["cost"] == 0.1
    assert "completion_time" not in response["usage"]
    # Throughput uses the count paired with the nested timing measurement.
    assert provider_output_speed(response) == 100


@pytest.mark.parametrize("canonical,nested", [
    ({"completion_tokens": 4}, {"completion_tokens": -1, "completion_time": 0.02}),
    ({"completion_tokens": 4}, {"completion_time": 0.02}),
    ({"completion_time": 0.02}, {"completion_tokens": 4}),
])
def test_groq_stream_does_not_pair_counts_and_time_from_different_reports(canonical, nested):
    response = stream_response([{
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": canonical,
        "x_groq": {"usage": nested},
    }])

    assert response["usage"] == canonical
    assert response["x_groq"]["usage"] == nested
    assert provider_output_speed(response) is None


@pytest.mark.parametrize("metrics", [
    {"timings": {"predicted_per_second": 200, "predicted_ms": 10}},
    {"stats": {"tokens_per_second": 200}},
    {"eval_count": 2, "eval_duration": 10_000_000},
])
def test_stream_preserves_metrics_in_final_usage_only_chunk(metrics):
    response = stream_response([
        {"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 2}, **metrics},
    ])

    assert all(response[key] == value for key, value in metrics.items())
    assert provider_output_speed(response) == 200
