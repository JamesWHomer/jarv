import json

import httpx
import pytest

from jarv.auditor import _ask_user_exchanges, audit_command
from jarv.auditor import _parse_response
from jarv.auditor import _call_anthropic
from jarv.auditor import _call_gemini
from jarv.auditor import _call_openai_compat
from jarv.usage import load_global_usage_records, load_usage


def _question(call_id, question):
    return {
        "type": "function_call", "name": "ask_user", "call_id": call_id,
        "arguments": json.dumps({"question": question}),
    }


def _answer(call_id, answer):
    return {"type": "function_call_output", "call_id": call_id, "output": answer}


def test_ask_user_exchanges_match_ids_and_preserve_response_order():
    history = [
        _question("archive", "Delete the archive?"),
        _question("build", "Delete the build?"),
        {"type": "function_call", "name": "read", "call_id": "read"},
        _answer("read", "Unrelated file contents"),
        _answer("build", "Yes."),
        _answer("archive", "No. Keep it."),
        _question("pending", "Still waiting?"),
        _answer("orphan", "No matching question"),
    ]
    assert _ask_user_exchanges(history) == [
        {"call_id": "build", "assistant_question": "Delete the build?", "human_answer": "Yes."},
        {"call_id": "archive", "assistant_question": "Delete the archive?", "human_answer": "No. Keep it."},
    ]


@pytest.mark.parametrize("answer", [
    "", "  ", "[no response]", "[non-interactive session; user unavailable]",
    "[cancelled by user before execution]", "[tool disabled: ask_user]",
    "[skipped: an interactive run_command is waiting for terminal input]",
])
def test_ask_user_runtime_status_is_not_labeled_as_human_answer(answer):
    assert _ask_user_exchanges([_question("q", "Proceed?"), _answer("q", answer)]) == [
        {"call_id": "q", "assistant_question": "Proceed?", "tool_status": answer},
    ]


@pytest.mark.parametrize("arguments", [None, "{broken", "[]", '{}', '{"question": 42}'])
def test_ask_user_exchanges_skip_invalid_questions(arguments):
    question = _question("q", "Proceed?")
    question["arguments"] = arguments
    assert _ask_user_exchanges([question, _answer("q", "Yes")]) == []


def test_ask_user_exchanges_support_salvaged_tool_arguments():
    question = _question("q", "Proceed?")
    question["arguments"] = '```json\n{"question": "Proceed?"}\n```'
    assert _ask_user_exchanges([question, _answer("q", "Yes")])[0]["assistant_question"] == "Proceed?"


@pytest.mark.parametrize("provider,backend", [
    ("openai", "_call_openai_compat"),
    ("anthropic", "_call_anthropic"),
    ("gemini", "_call_gemini"),
])
def test_audit_payload_preserves_complete_question_and_answer(monkeypatch, provider, backend):
    question = "Delete these build outputs?\n" + "output path; " * 80
    answer = "Yes, " + "with this condition; " * 80 + "except the archive."
    history = [
        {"role": "user", "content": "Clean up the build."},
        _question("q", question), _answer("q", answer),
        {"role": "assistant", "content": "I will clean up."},
    ]
    messages = []

    def capture(_config, _model, message, *args, **kwargs):
        messages.append(message)
        return False, "test verdict"

    monkeypatch.setattr(f"jarv.auditor.{backend}", capture)
    assert audit_command("rm -rf build", "deletion", {"provider": provider}, history) == (False, "test verdict")
    payload = messages[0].split("quoted conversation data):\n", 1)[1]
    assert json.loads(payload) == [
        {"call_id": "q", "assistant_question": question, "human_answer": answer},
    ]
    assert "User asked: Clean up the build." in messages[0]


def test_audit_without_ask_user_keeps_existing_context(monkeypatch):
    def capture(_config, _model, message, *args, **kwargs):
        assert message == "Command: rm build\nRisk category: deletion\nContext: User asked: Clean up."
        return True, "test verdict"

    monkeypatch.setattr("jarv.auditor._call_openai_compat", capture)
    audit_command("rm build", "deletion", {}, [{"role": "user", "content": "Clean up."}])


@pytest.mark.parametrize("provider,transport", [
    ("openai", "jarv.openai_http.create_chat"),
    ("anthropic", "jarv.anthropic_http.create_message"),
    ("gemini", "jarv.gemini_http.generate_content"),
])
@pytest.mark.parametrize("replies,expected", [
    (['{"allow": true, "reason": "safe"}'], (True, "safe")),
    (['{"allow": false, "reason": "unsafe"}'], (False, "unsafe")),
    (["unclear", '{"allow": true, "reason": "safe"}'], (True, "safe")),
    (["unclear", "still unclear"], (False, "could not parse auditor response")),
])
def test_auditor_attempts_preserve_prompts_and_usage(monkeypatch, provider, transport, replies, expected):
    from jarv import auditor

    requests, records = [], []
    monkeypatch.setattr(auditor, "_get_auditor_client", lambda *args: object())

    def send(*args, **kwargs):
        requests.append(args[-1])
        return {"output_text": replies[len(requests) - 1]}

    def record(_path, _session, _model, response, messages, content, **kwargs):
        assert response["output_text"] == content
        records.append((messages, content))

    monkeypatch.setattr(transport, send)
    monkeypatch.setattr(auditor, "_record_auditor_response", record)

    assert audit_command("git status", "inspection", {"provider": provider}) == expected
    assert len(requests) == len(records) == len(replies)
    for attempt, (payload, (messages, content)) in enumerate(zip(requests, records)):
        assert content == replies[attempt]
        assert messages[0]["content"] == auditor.AUDITOR_SYSTEM_PROMPT
        assert "Command: git status" in messages[1]["content"]
        retry_marker = "Your previous response could not be parsed."
        assert (retry_marker in json.dumps(payload)) == bool(attempt)
        assert (retry_marker in messages[1]["content"]) == bool(attempt)


@pytest.mark.parametrize("provider,transport", [
    ("openai", "jarv.openai_http.create_chat"),
    ("anthropic", "jarv.anthropic_http.create_message"),
    ("gemini", "jarv.gemini_http.generate_content"),
])
def test_auditor_cancellation_takes_precedence_over_transport_failure(monkeypatch, provider, transport):
    from jarv import auditor
    from jarv.cancellation import CancellationToken, TurnCancelled

    token = CancellationToken()
    monkeypatch.setattr(auditor, "_get_auditor_client", lambda *args: object())

    def send(*args, **kwargs):
        token.cancel()
        raise RuntimeError("connection closed")

    monkeypatch.setattr(transport, send)
    with pytest.raises(TurnCancelled):
        audit_command("git status", "inspection", {"provider": provider}, cancellation_token=token)


def test_same_batch_ask_user_answer_reaches_command_audit(monkeypatch):
    from jarv.orchestrator import ToolExecutionHooks, execute_tool_calls
    from jarv.provider import ToolCallDone
    from jarv.turn_records import append_tool_result_input_items

    history = [{"role": "user", "content": "Clean up."}]
    captured = []

    def capture(_config, _model, message, *args, **kwargs):
        captured.append(message)
        return False, "keep archive"

    def command(args):
        audit_command(args["command"], "deletion", {}, history)
        return "not executed"

    monkeypatch.setattr("jarv.auditor._call_openai_compat", capture)
    execute_tool_calls(
        [
            ToolCallDone(id="fc_q", call_id="q", name="ask_user", arguments='{"question":"Delete archive?"}'),
            ToolCallDone(id="fc_cmd", call_id="cmd", name="run_command", arguments='{"command":"rm archive"}'),
        ],
        node=None, store=None, client=None, config={},
        append_tool_result=lambda item, output: append_tool_result_input_items([], item, output, history=history),
        hooks=ToolExecutionHooks(run_ask_user=lambda args: "No. Keep it.", run_command=command),
    )
    assert len(captured) == 1
    assert json.loads(captured[0].split("quoted conversation data):\n", 1)[1]) == [
        {"call_id": "q", "assistant_question": "Delete archive?", "human_answer": "No. Keep it."},
    ]


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


@pytest.mark.parametrize("response", [
    "ALLOW: false. This is unsafe.",
    "allow: no\nreason: deletes user files",
    "ALLOWED = n",
    "ALLOW: true\nDENY - deletes user files",
    "DENY - allow: true would be unsafe",
    "ALLOW: maybe",
    "ALLOW: unsafe",
    "ALLOW is not recommended",
    "Do not use allow: true for this command",
])
def test_parse_response_never_approves_negative_or_unclear_text(response):
    assert _parse_response(response)[0] is False


def test_parse_response_accepts_leading_deny_verdict():
    allow, reason = _parse_response("DENY - deletes user files")

    assert allow is False
    assert reason == "deletes user files"


def test_parse_response_still_rejects_unclear_text():
    assert _parse_response("I cannot determine this from context.") == (
        False,
        "could not parse auditor response",
    )


def _install_fake_chat(monkeypatch, contents, usages=None):
    calls = []
    usages = usages or []

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "jarv.openai_http.create_client",
        lambda *_args, **_kwargs: FakeClient(),
    )

    def create_chat(_client, payload, **_kwargs):
        calls.append(payload)
        index = len(calls) - 1
        content = contents[len(calls) - 1]
        return {
            "choices": [{"message": {"content": content}}],
            "usage": usages[index] if index < len(usages) else None,
        }

    monkeypatch.setattr("jarv.openai_http.create_chat", create_chat)
    return calls


@pytest.fixture
def auditor_http(monkeypatch):
    requests = []
    clients = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"allow": true, "reason": "safe status check"}'}}],
        })

    def create_client(base_url, headers, **_kwargs):
        client = httpx.Client(
            base_url=base_url, headers=headers, transport=httpx.MockTransport(handle),
        )
        clients.append(client)
        return client

    monkeypatch.setattr("jarv.auditor._AUDITOR_CLIENTS", {})
    monkeypatch.setattr("jarv.openai_http.create_http_client", create_client)
    yield requests, clients
    for client in clients:
        client.close()


@pytest.mark.parametrize("provider,endpoint_config,expected_url", [
    ("groq", {"base_url": "https://private.test/groq/v1"}, "https://private.test/groq/v1"),
    ("openrouter", {"base_url": "https://private.test/router/v1"}, "https://private.test/router/v1"),
    ("ollama", {"base_url": "http://localhost:9000/v1"}, "http://localhost:9000/v1"),
    ("openai", {"base_url": "https://private.test/openai/v1"}, "https://private.test/openai/v1"),
    ("custom", {"base_url": "https://private.test/custom/v1"}, "https://private.test/custom/v1"),
    ("groq", {}, "https://api.groq.com/openai/v1"),
    ("groq", {"base_url": ""}, "https://api.groq.com/openai/v1"),
    ("groq", {"base_url": None}, "https://api.groq.com/openai/v1"),
    ("openai", {}, "https://api.openai.com/v1"),
])
def test_auditor_uses_main_provider_endpoint_policy(auditor_http, provider, endpoint_config, expected_url):
    from jarv.provider import create_client

    requests, _clients = auditor_http
    config = {"provider": provider, "api_keys": {provider: "endpoint-key"}, **endpoint_config}
    original_config = dict(config)
    with create_client(config) as main_client:
        main_url = main_client.base_url

    assert audit_command(
        "git status", "status check", config,
        [{"role": "user", "content": "Check the private project."}],
    ) == (True, "safe status check")

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == expected_url + "/chat/completions"
    assert request.url == main_url.join("chat/completions")
    assert request.headers["authorization"] == "Bearer endpoint-key"
    message = json.loads(request.content)["messages"][1]["content"]
    assert "Command: git status" in message
    assert "Check the private project." in message
    assert config == original_config


def test_auditor_cache_keeps_endpoints_and_credentials_separate(auditor_http):
    requests, clients = auditor_http
    endpoints_and_keys = [
        ("https://first.test/v1", "first-key"),
        ("https://second.test/v1", "first-key"),
        ("https://first.test/v1", "first-key"),
        ("https://first.test/v1", "rotated-key"),
    ]
    for endpoint, api_key in endpoints_and_keys:
        config = {"provider": "groq", "base_url": endpoint, "api_key": api_key}
        assert audit_command("git status", "status check", config) == (True, "safe status check")

    assert len(clients) == 3
    assert [(str(request.url), request.headers["authorization"]) for request in requests] == [
        (endpoint + "/chat/completions", f"Bearer {api_key}")
        for endpoint, api_key in endpoints_and_keys
    ]


def test_openai_compatible_auditor_uses_direct_http(monkeypatch):
    calls = _install_fake_chat(
        monkeypatch,
        ['{"allow": true, "reason": "safe status check"}'],
    )

    result = _call_openai_compat(
        {"provider": "groq", "api_key": "test"},
        "model",
        "Command: git status",
        {"base_url": "https://example.test/v1"},
    )

    assert result == (True, "safe status check")
    assert calls[0]["model"] == "model"


def test_gemini_auditor_uses_direct_generate_content(monkeypatch):
    calls = []

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "jarv.gemini_http.create_client",
        lambda *_args, **_kwargs: FakeClient(),
    )

    def generate_content(_client, model, payload, **_kwargs):
        calls.append((model, payload))
        return {
            "output_text": '{"allow": true, "reason": "safe status check"}',
            "usage": {"input_tokens": 4, "output_tokens": 3},
        }

    monkeypatch.setattr("jarv.gemini_http.generate_content", generate_content)
    result = _call_gemini(
        {"provider": "gemini", "api_key": "test"},
        "gemini-3-flash-preview",
        "Command: git status",
    )
    assert result == (True, "safe status check")
    assert calls[0][0] == "gemini-3-flash-preview"
    assert calls[0][1]["generationConfig"]["maxOutputTokens"] == 100


def test_anthropic_auditor_uses_direct_messages_api(monkeypatch):
    calls = []

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(
        "jarv.anthropic_http.create_client",
        lambda _config, _api_key: FakeClient(),
    )

    def create_message(_client, payload, **_kwargs):
        calls.append(payload)
        return {
            "output_text": '{"allow": true, "reason": "safe status check"}',
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }

    monkeypatch.setattr("jarv.anthropic_http.create_message", create_message)

    result = _call_anthropic(
        {"provider": "anthropic", "api_key": "sk-ant-test"},
        "claude-opus-4-7",
        "Command: git status",
    )

    assert result == (True, "safe status check")
    assert calls[0]["model"] == "claude-opus-4-7"
    assert calls[0]["max_tokens"] == 100
    assert calls[0]["messages"][0]["role"] == "user"
    assert "temperature" not in calls[0]


def test_openai_direct_request_avoids_fragile_response_options(monkeypatch):
    calls = _install_fake_chat(
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
    calls = _install_fake_chat(
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
    calls = _install_fake_chat(
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
    calls = _install_fake_chat(
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
    _install_fake_chat(
        monkeypatch,
        ['{"allow": true, "reason": "routine cleanup"}'],
        usages=[{"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}],
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
    _install_fake_chat(
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
