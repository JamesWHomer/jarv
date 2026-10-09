"""Per-response usage is factual about token counts and the timing denominator."""

from types import SimpleNamespace

import pytest

from jarv.turn_stats import TurnSummaryStats, format_turn_summary
from jarv.turn_loop import StreamCollection


CORE_FIELDS = {"tokens", "cache", "reasoning", "time", "speed"}


def stats(response, *, seconds=2.0, fields=CORE_FIELDS, **kwargs):
    return format_turn_summary(
        response, model="test-model", elapsed_seconds=seconds, fields=fields, **kwargs,
    ).plain


def text_stream(**overrides):
    values = dict(
        reply_text="abcdefghijklmnop", streamed_text="abcdefghijklmnop",
        first_text_chunk="abcd", text_stream_seconds=0.5, text_chunk_count=3,
    )
    values.update(overrides)
    return StreamCollection(**values)


@pytest.mark.parametrize("request_seconds", [2.0, 100.0, None])
def test_stream_speed_excludes_wait_reasoning_and_first_chunk(request_seconds):
    line = stats({"usage": {
        "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 10},
    }}, seconds=request_seconds, fields={"speed"},
        stream_result=text_stream(saw_reasoning=True))

    # 10 visible tokens, 1/4 in the first chunk; 7.5 tokens over 0.5s.
    assert line == "Turn: ~15.0 tok/s (stream)"


@pytest.mark.parametrize("overrides", [
    {"text_chunk_count": 1}, {"text_stream_seconds": 0},
    {"text_stream_seconds": 0.01}, {"text_stream_seconds": float("nan")},
    {"text_stream_seconds": None}, {"first_text_chunk": ""},
    {"streamed_text": ""}, {"reply_text": "recovered final text"},
    {"final_text": "short final text"},
    {"tool_calls": [object()]}, {"saw_reasoning": True},
])
def test_unreliable_stream_windows_do_not_show_a_numeric_rate(overrides):
    line = stats({"usage": {"output_tokens": 20}}, fields={"speed"},
        stream_result=text_stream(**overrides))

    assert line == "Turn: tok/s unavailable"


def test_fractional_token_window_is_too_short_to_measure():
    line = stats({"usage": {"output_tokens": 1}}, fields={"speed"},
        stream_result=text_stream())

    assert line == "Turn: tok/s unavailable"


def test_oversized_output_count_does_not_crash_stream_speed():
    line = stats({"usage": {"output_tokens": 10 ** 400}}, fields={"speed"},
        stream_result=text_stream())

    assert line == "Turn: tok/s unavailable"


def test_server_speed_takes_precedence_for_short_or_tool_responses():
    line = stats({"usage": {
        "completion_tokens": 3, "completion_time": 0.02,
    }}, seconds=10, fields={"speed"}, stream_result=text_stream(
        text_chunk_count=1, tool_calls=[object()],
    ))

    assert line == "Turn: 150.0 tok/s (server)"


def test_server_speed_does_not_depend_on_estimated_text_tokens():
    line = stats({"timings": {"predicted_per_second": 100}},
        fields={"speed"}, output_text="abcdefgh")

    assert line == "Turn: 100.0 tok/s (server)"


def test_stats_keeps_request_time_separate_from_generation_speed():
    line = stats({"usage": {
        "input_tokens": 1_200,
        "input_tokens_details": {"cached_tokens": 800},
        "output_tokens": 60,
        "output_tokens_details": {"reasoning_tokens": 40},
        "total_tokens": 1_260,
    }}, output_text="hi")

    assert "1,200 in" in line
    assert "800 cached" in line
    assert "60 out" in line
    assert "40 reasoning" in line
    assert "1,260 total" in line
    assert "2.00s" in line
    assert "tok/s unavailable" in line
    assert "estimated" not in line


def test_stats_accepts_chat_completion_usage_objects():
    line = stats(SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=12,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
    )))

    assert "100 in" in line
    assert "12 out" in line
    assert "4 reasoning" in line
    assert "112 total" in line
    assert "tok/s unavailable" in line


@pytest.mark.parametrize("reported_total", [None, 180])
def test_gemini_counts_include_thoughts_without_double_counting_total(reported_total):
    usage = {"input_tokens": 100, "output_tokens": 20, "reasoning_output_tokens": 60}
    if reported_total is not None:
        usage["total_tokens"] = reported_total

    line = stats({"usage": usage}, provider="gemini")

    assert "80 out" in line
    assert "60 reasoning" in line
    assert "180 total" in line
    assert "tok/s unavailable" in line


def test_missing_usage_labels_estimated_counts_without_inventing_speed():
    line = stats({}, context_breakdown={"system": 12, "history": 8}, output_text="abcdefgh")

    assert "20 in" in line
    assert "2 out" in line
    assert "22 total" in line
    assert "tok/s unavailable" in line
    assert "usage estimated" in line


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
@pytest.mark.parametrize("missing_usage", [None, {}])
def test_native_adapter_missing_usage_falls_back_to_estimates(provider, missing_usage):
    if provider == "anthropic":
        from jarv.anthropic_http import normalize_response
        response = normalize_response({"usage": missing_usage})
    else:
        from jarv.gemini_http import normalize_response
        response = normalize_response({"usageMetadata": missing_usage})

    line = stats(
        response, provider=provider, context_breakdown={"history": 20},
        output_text="abcdefgh",
    )

    assert response["usage"] == {}
    assert "20 in" in line
    assert "2 out" in line
    assert "tok/s unavailable" in line
    assert "usage estimated" in line


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_native_adapter_explicit_zero_usage_remains_measured(provider):
    if provider == "anthropic":
        from jarv.anthropic_http import normalize_response
        response = normalize_response({"usage": {"input_tokens": 0, "output_tokens": 0}})
    else:
        from jarv.gemini_http import normalize_response
        response = normalize_response({"usageMetadata": {
            "promptTokenCount": 0, "candidatesTokenCount": 0,
        }})

    line = stats(
        response, provider=provider, context_breakdown={"history": 20},
        output_text="abcdefgh",
    )

    assert "0 in" in line
    assert "0 out" in line
    assert "tok/s unavailable" in line
    assert "estimated" not in line


def test_missing_usage_does_not_invent_token_counts_or_a_rate():
    line = stats({})

    assert "token usage unavailable" in line
    assert "tok/s unavailable" in line
    assert "2.00s" in line
    assert "estimated" not in line


def test_partial_usage_does_not_treat_missing_output_as_measured_zero():
    line = stats({"usage": {"input_tokens": 10}})

    assert "10 in" in line
    assert "? out" in line
    assert "? total" in line
    assert "tok/s unavailable" in line


def test_partial_output_usage_can_use_server_rate_without_inventing_input():
    line = stats({"usage": {"completion_tokens": 10, "completion_time": 2.0}})

    assert "? in" in line
    assert "10 out" in line
    assert "? total" in line
    assert "5.0 tok/s (server)" in line


def test_partial_estimate_marks_missing_input_as_unknown():
    line = stats({}, output_text="abcdefgh")

    assert "? in" in line
    assert "2 out" in line
    assert "? total" in line
    assert "usage estimated" in line


@pytest.mark.parametrize("seconds", [0, -1, None, True, float("nan"), float("inf")])
def test_invalid_duration_never_produces_a_misleading_rate(seconds):
    line = stats({"usage": {"input_tokens": 2, "output_tokens": 4}}, seconds=seconds)

    assert "2 in" in line
    assert "4 out" in line
    assert "tok/s unavailable" in line
    assert "avg output tok/s" not in line


def test_zero_output_without_server_timing_has_no_generation_rate():
    line = stats({"usage": {"input_tokens": 2, "output_tokens": 0}})

    assert "0 out" in line
    assert "tok/s unavailable" in line


@pytest.mark.parametrize("field,expected", [
    ("tokens", "100 in · 20 out · 120 total"),
    ("cache", "30 cached"),
    ("reasoning", "10 reasoning"),
    ("time", "2.00s"),
    ("speed", "tok/s unavailable"),
    ("session", "800 session tokens"),
    ("cost", "session cost $0.050"),
])
def test_fields_can_be_shown_independently(field, expected):
    line = stats({"usage": {
        "input_tokens": 100, "output_tokens": 20,
        "cached_input_tokens": 30, "reasoning_output_tokens": 10,
    }}, fields={field}, session_usage={"totals": {
        "total_tokens": 800, "provider_cost_usd": 0.05,
        "cost_exact_request_count": 3,
    }})

    assert line == f"Turn: {expected}"


@pytest.mark.parametrize("hidden,fragment", [
    ("tokens", " in"), ("cache", " cached"), ("reasoning", " reasoning"),
    ("time", "2.00s"), ("speed", "tok/s"),
])
def test_disabled_fields_are_absent(hidden, fragment):
    line = stats({"usage": {
        "input_tokens": 100, "output_tokens": 20,
        "cached_input_tokens": 30, "reasoning_output_tokens": 10,
    }}, fields=CORE_FIELDS - {hidden})

    assert fragment not in line


def test_gemini_stream_speed_excludes_thoughts():
    line = stats({"usage": {
        "input_tokens": 100, "output_tokens": 20, "reasoning_output_tokens": 60,
    }}, provider="gemini", fields={"speed"}, stream_result=text_stream(saw_reasoning=True))

    assert line == "Turn: ~30.0 tok/s (stream)"


@pytest.mark.parametrize("fields", [set(), {"unknown_field"}])
def test_no_selected_fields_emit_no_summary(fields):
    assert format_turn_summary({}, model="test", elapsed_seconds=2, fields=fields) is None


def test_missing_token_details_are_not_fabricated():
    assert format_turn_summary(
        {"usage": {"input_tokens": 100, "output_tokens": 20}},
        model="test", elapsed_seconds=2, fields={"cache", "reasoning"},
    ) is None


def test_reported_zero_details_are_displayed():
    line = stats({"usage": {
        "input_tokens": 100, "output_tokens": 20,
        "cached_input_tokens": 0, "reasoning_output_tokens": 0,
    }}, fields={"cache", "reasoning"})

    assert line == "Turn: 0 cached · 0 reasoning"


def test_session_fields_are_explicitly_unavailable_without_saved_usage():
    line = stats({}, fields={"session", "cost"})

    assert line == "Turn: session usage unavailable · session cost unavailable"


def test_time_only_does_not_expose_estimated_or_unavailable_tokens():
    line = stats({}, fields={"time"}, output_text="abcdefgh")

    assert line == "Turn: 2.00s"


def test_heuristic_token_counts_are_not_presented_as_generation_speed():
    line = stats({}, fields={"speed"}, output_text="abcdefgh")

    assert line == "Turn: tok/s unavailable"


def test_session_summary_does_not_inherit_current_response_estimate_label():
    line = stats({}, fields={"session"}, output_text="abcdefgh", session_usage={
        "totals": {"total_tokens": 800},
    })

    assert line == "Turn: 800 session tokens"


def test_estimated_session_cost_keeps_incomplete_status():
    line = stats({}, fields={"cost"}, session_usage={"totals": {
        "estimated_cost_usd": 0.05,
        "cost_estimated_request_count": 1,
        "cost_unknown_request_count": 1,
    }})

    assert line == "Turn: est. session cost $0.050 (incomplete)"


def test_actual_zero_cost_is_not_unavailable():
    line = stats({}, fields={"cost"}, session_usage={"totals": {
        "provider_cost_usd": 0, "cost_exact_request_count": 1,
    }})

    assert line == "Turn: session cost $0.00"


def accumulate(*responses):
    turn = TurnSummaryStats()
    for response, options in responses:
        options = dict(options)
        turn.add_response(
            response, model="test-model", elapsed_seconds=options.pop("elapsed_seconds", 2.0),
            **options,
        )
    return turn


def test_turn_accumulates_all_model_requests_but_reports_only_final_response_speed():
    turn = accumulate(
        ({"usage": {
            "input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 30,
            "reasoning_output_tokens": 5,
        }, "timings": {"predicted_per_second": 800}}, {"elapsed_seconds": 2.0}),
        ({"usage": {
            "prompt_tokens": 200, "completion_tokens": 10, "completion_time": 0.05,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }}, {"elapsed_seconds": 3.0}),
    )

    line = stats({}, seconds=999, turn_stats=turn)

    assert turn.request_count == 2
    assert turn.elapsed_seconds == 5.0
    assert "300 in · 30 out · 330 total" in line
    assert "90 cached · 7 reasoning" in line
    assert "5.00s" in line
    assert "final response 200.0 tok/s (server)" in line
    assert "800.0" not in line
    assert "999" not in line
    assert "estimated" not in line


def test_aggregated_gemini_thoughts_are_counted_once_and_final_stream_stays_separate():
    turn = accumulate(
        ({"usage": {
            "input_tokens": 100, "output_tokens": 20, "reasoning_output_tokens": 60,
        }}, {"provider": "gemini"}),
        ({"usage": {
            "input_tokens": 50, "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 10},
        }}, {"provider": "openai", "stream_result": text_stream(saw_reasoning=True)}),
    )

    line = stats({}, turn_stats=turn)

    assert "150 in · 120 out · 270 total" in line
    assert "70 reasoning" in line
    # Final response: 30 visible tokens, minus first 1/4, over 0.5 seconds.
    assert "final response ~45.0 tok/s (stream)" in line
    assert stats({}, turn_stats=turn) == line


def test_missing_response_usage_makes_aggregate_counts_unknown():
    turn = accumulate(
        ({"usage": {
            "input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 30,
            "reasoning_output_tokens": 5,
        }}, {}),
        ({}, {}),
    )

    line = stats({}, fields={"tokens", "cache", "reasoning"}, turn_stats=turn)

    assert line == "Turn: ? in · ? out · ? total · ? cached · ? reasoning"


def test_partially_reported_response_preserves_only_complete_aggregate_fields():
    turn = accumulate(
        ({"usage": {"input_tokens": 100, "output_tokens": 20}}, {}),
        ({"usage": {"input_tokens": 200}}, {}),
    )

    assert stats({}, fields={"tokens"}, turn_stats=turn) == "Turn: 300 in · ? out · ? total"


@pytest.mark.parametrize("estimate_first", [False, True])
def test_estimate_in_any_response_labels_the_aggregate(estimate_first):
    measured = ({"usage": {"input_tokens": 100, "output_tokens": 20}}, {})
    estimate = ({}, {"context_breakdown": {"history": 8}, "output_text": "abcdefgh"})
    turn = accumulate(*([estimate, measured] if estimate_first else [measured, estimate]))

    assert stats({}, fields={"tokens"}, turn_stats=turn) == (
        "Turn: 108 in · 22 out · 130 total · usage estimated"
    )
    assert stats({}, fields={"time"}, turn_stats=turn) == "Turn: 4.00s"


def test_no_reported_or_estimated_counts_remain_unavailable():
    turn = accumulate(({}, {}), ({}, {}))

    assert stats({}, fields={"tokens"}, turn_stats=turn) == "Turn: token usage unavailable"
    assert format_turn_summary(
        {}, model="test-model", elapsed_seconds=1.0,
        fields={"cache", "reasoning"}, turn_stats=turn,
    ) is None


@pytest.mark.parametrize("invalid", [None, True, -1, float("nan"), float("inf")])
def test_unknown_request_time_makes_aggregate_time_unavailable(invalid):
    turn = accumulate(({}, {"elapsed_seconds": 2.0}), ({}, {"elapsed_seconds": invalid}))

    assert stats({}, fields={"time"}, turn_stats=turn) == "Turn: time unavailable"


def test_zero_duration_does_not_discard_other_measured_request_time():
    turn = accumulate(({}, {"elapsed_seconds": 2.0}), ({}, {"elapsed_seconds": 0.0}))

    assert stats({}, fields={"time"}, turn_stats=turn) == "Turn: 2.00s"


def test_overflowing_aggregate_duration_is_unavailable():
    turn = accumulate(({}, {"elapsed_seconds": 1e308}), ({}, {"elapsed_seconds": 1e308}))

    assert stats({}, fields={"time"}, turn_stats=turn) == "Turn: time unavailable"


def test_missing_final_speed_does_not_fall_back_to_a_previous_request():
    turn = accumulate(
        ({"timings": {"predicted_per_second": 800}}, {}),
        ({"usage": {"output_tokens": 20}}, {"stream_result": text_stream(text_chunk_count=1)}),
    )

    assert stats({}, fields={"speed"}, turn_stats=turn) == "Turn: final response tok/s unavailable"


def test_single_response_accumulator_preserves_existing_summary_format():
    response = {"usage": {
        "input_tokens": 100, "output_tokens": 20, "reasoning_output_tokens": 5,
    }}
    stream = text_stream()
    turn = accumulate((response, {"stream_result": stream}))

    assert stats({}, turn_stats=turn) == stats(response, stream_result=stream)


def test_aggregate_summary_keeps_session_fields_independent():
    turn = accumulate(({"usage": {"input_tokens": 100, "output_tokens": 20}}, {}), ({}, {}))

    line = stats({}, fields={"session", "cost"}, turn_stats=turn, session_usage={"totals": {
        "total_tokens": 1500, "provider_cost_usd": 0.05, "cost_exact_request_count": 3,
    }})

    assert line == "Turn: 1,500 session tokens · session cost $0.050"


def test_disabled_aggregate_fields_emit_no_line():
    turn = accumulate(({"usage": {"input_tokens": 100, "output_tokens": 20}}, {}))

    assert format_turn_summary(
        {}, model="test-model", elapsed_seconds=1.0, fields=set(), turn_stats=turn,
    ) is None
