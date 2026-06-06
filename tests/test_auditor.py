import sys
from types import SimpleNamespace

from jarv.auditor import _parse_response
from jarv.auditor import _call_litellm
from jarv.auditor import _call_openai_compat
from jarv.usage import load_global_usage_records, load_usage


def test_parse_response_accepts_strict_json():
    assert _parse_response('{"allow": true, "reason": "routine install"}') == (
        True,
        "routine install",
    )


def test_parse_response_accepts_json_with_surrounding_text():
    assert _parse_response('Sure.\n{"allow": true, "reason": "safe in context"}') == (
        True,
        "safe in context",
    )


def test_parse_response_accepts_json_after_powershell_block_text():
    assert _parse_response(
        'The command includes if (-not $?) { exit 1 }.\n'
        '{"allow": true, "reason": "only removes a temp file"}'
    ) == (
        True,
        "only removes a temp file",
    )


def test_parse_response_accepts_json_before_powershell_block_text():
    assert _parse_response(
        '{"allow": true, "reason": "safe tag recreation"}\n'
        "Note: cleanup uses if (-not $?) { exit 1 }."
    ) == (
        True,
        "safe tag recreation",
    )


def test_parse_response_skips_invalid_braces_before_json():
    assert _parse_response(
        'First brace is not JSON: { exit 1 }.\n'
        '{"allow": "false", "reason": "tag deletion needs review"}'
    ) == (
        False,
        "tag deletion needs review",
    )


def test_parse_response_accepts_fenced_json_with_surrounding_text():
    assert _parse_response(
        'Verdict:\n```json\n{"allow": "true", "reason": "temp file cleanup only"}\n```\nDone.'
    ) == (
        True,
        "temp file cleanup only",
    )


def test_parse_response_accepts_json_reason_containing_braces():
    assert _parse_response(
        '{"allow": true, "reason": "PowerShell block { exit 1 } is only error handling"}'
    ) == (
        True,
        "PowerShell block { exit 1 } is only error handling",
    )


def test_parse_response_ignores_json_without_allow_key():
    assert _parse_response(
        '{"reason": "missing verdict"}\n{"allow": true, "reason": "safe cleanup"}'
    ) == (
        True,
        "safe cleanup",
    )


def test_parse_response_accepts_json_verdict_alias():
    assert _parse_response('{"verdict": "deny", "reason": "removes user files"}') == (
        False,
        "removes user files",
    )


def test_parse_response_accepts_json_decision_alias():
    assert _parse_response('{"decision": "approve", "reason": "version probe only"}') == (
        True,
        "version probe only",
    )


def test_parse_response_rejects_unclear_json_allow_value():
    assert _parse_response('{"allow": "maybe", "reason": "ambiguous"}') == (
        False,
        "could not parse auditor response",
    )


def test_parse_response_accepts_allow_key_value_text():
    assert _parse_response("allow: true\nreason: version probe only") == (
        True,
        "version probe only",
    )


def test_parse_response_accepts_leading_allow_verdict():
    assert _parse_response("ALLOW - harmless package resolution check")[0] is True


def test_parse_response_accepts_leading_deny_verdict():
    allow, reason = _parse_response("DENY - deletes user files")

    assert allow is False
    assert reason == "deletes user files"


def test_parse_response_still_rejects_unclear_text():
    assert _parse_response("I cannot determine this from context.") == (
        False,
        "could not parse auditor response",
    )


def _install_fake_openai(monkeypatch, contents, usages=None):
    calls = []
    usages = usages or []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            index = len(calls) - 1
            content = contents[len(calls) - 1]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=usages[index] if index < len(usages) else None,
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    return calls


def _install_fake_litellm(monkeypatch, contents):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        content = contents[len(calls) - 1]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=None,
        )

    fake = SimpleNamespace(completion=completion)
    monkeypatch.setattr("jarv.litellm_compat.import_litellm", lambda: fake)
    return calls


def test_litellm_auditor_omits_temperature(monkeypatch):
    calls = _install_fake_litellm(
        monkeypatch,
        ['{"allow": true, "reason": "safe status check"}'],
    )

    result = _call_litellm(
        {"provider": "anthropic"},
        "claude-opus-4-7",
        "Command: git status",
    )

    assert result == (True, "safe status check")
    assert calls[0]["model"] == "anthropic/claude-opus-4-7"
    assert "temperature" not in calls[0]


def test_litellm_auditor_retry_also_omits_temperature(monkeypatch):
    calls = _install_fake_litellm(
        monkeypatch,
        [
            "I cannot determine this from context.",
            '{"allow": true, "reason": "safe version probe"}',
        ],
    )

    result = _call_litellm(
        {"provider": "anthropic"},
        "claude-opus-4-7",
        "Command: python --version",
    )

    assert result == (True, "safe version probe")
    assert len(calls) == 2
    assert all("temperature" not in call for call in calls)


def test_non_anthropic_litellm_auditor_omits_temperature(monkeypatch):
    calls = _install_fake_litellm(
        monkeypatch,
        ['{"allow": true, "reason": "safe status check"}'],
    )

    _call_litellm(
        {"provider": "gemini"},
        "gemini-3-flash-preview",
        "Command: git status",
    )

    assert calls[0]["model"] == "gemini/gemini-3-flash-preview"
    assert "temperature" not in calls[0]


def test_openai_direct_request_avoids_fragile_response_options(monkeypatch):
    calls = _install_fake_openai(
        monkeypatch,
        ['{"allow": true, "reason": "routine cleanup"}'],
    )

    result = _call_openai_compat(
        {"provider": "openai"},
        "gpt-4.1-mini",
        "Command: Remove-Item .cache",
        {"base_url": None},
    )

    assert result == (True, "routine cleanup")
    assert "response_format" not in calls[0]
    assert "reasoning_effort" not in calls[0]


def test_openai_reasoning_model_uses_larger_budget_without_extra_options(monkeypatch):
    calls = _install_fake_openai(
        monkeypatch,
        ['{"allow": true, "reason": "safe status check"}'],
    )

    result = _call_openai_compat(
        {"provider": "openai"},
        "gpt-5.4-mini",
        "Command: git status",
        {"base_url": None},
    )

    assert result == (True, "safe status check")
    assert calls[0]["max_completion_tokens"] == 300
    assert "max_tokens" not in calls[0]
    assert "response_format" not in calls[0]
    assert "reasoning_effort" not in calls[0]


def test_openai_unparsable_response_retries_once(monkeypatch):
    calls = _install_fake_openai(
        monkeypatch,
        [
            "I cannot determine this from context.",
            '{"allow": true, "reason": "safe version probe"}',
        ],
    )

    result = _call_openai_compat(
        {"provider": "openai"},
        "gpt-4.1-mini",
        "Command: python --version",
        {"base_url": None},
    )

    assert result == (True, "safe version probe")
    assert len(calls) == 2
    assert "previous response could not be parsed" in calls[1]["messages"][1][
        "content"
    ]


def test_openai_retry_failure_fails_closed(monkeypatch):
    calls = _install_fake_openai(
        monkeypatch,
        [
            "I cannot determine this from context.",
            "Still unclear.",
        ],
    )

    result = _call_openai_compat(
        {"provider": "openai"},
        "gpt-4.1-mini",
        "Command: Remove-Item important",
        {"base_url": None},
    )

    assert result == (False, "could not parse auditor response")
    assert len(calls) == 2


def test_openai_auditor_records_usage_metadata(monkeypatch, tmp_path):
    _install_fake_openai(
        monkeypatch,
        ['{"allow": true, "reason": "routine cleanup"}'],
        usages=[SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25)],
    )
    usage_path = tmp_path / "usage-test.json"
    global_path = tmp_path / "usage.json"

    result = _call_openai_compat(
        {"provider": "openai"},
        "test-model",
        "Command: Remove-Item .cache",
        {"base_url": None},
        usage_path=usage_path,
        session_id="session-id",
        global_usage_path=global_path,
    )

    session_usage = load_usage(usage_path, "session-id")
    global_records = load_global_usage_records(global_path)

    assert result == (True, "routine cleanup")
    assert session_usage["sources"]["auditor"]["request_count"] == 1
    assert global_records[0]["source"] == "auditor"
    assert global_records[0]["session_id"] == "session-id"
    assert global_records[0]["input_tokens"] == 20
    assert global_records[0]["output_tokens"] == 5


def test_openai_auditor_records_estimated_usage_when_provider_omits_usage(monkeypatch, tmp_path):
    _install_fake_openai(
        monkeypatch,
        ['{"allow": true, "reason": "safe version probe"}'],
    )
    usage_path = tmp_path / "usage-test.json"
    global_path = tmp_path / "usage.json"

    result = _call_openai_compat(
        {"provider": "openai"},
        "unknown-provider/model",
        "Command: python --version",
        {"base_url": None},
        usage_path=usage_path,
        session_id="session-id",
        global_usage_path=global_path,
    )

    global_records = load_global_usage_records(global_path)

    assert result == (True, "safe version probe")
    assert global_records[0]["source"] == "auditor"
    assert global_records[0]["estimated"] is True
    assert global_records[0]["input_tokens"] > 0
    assert global_records[0]["output_tokens"] > 0
