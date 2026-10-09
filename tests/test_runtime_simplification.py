"""Behavior at the stream and installer boundaries shared by refactored paths."""

import pytest

from jarv import anthropic_http, provider, standalone, uninstall
from jarv.cancellation import TurnCancelled


@pytest.mark.parametrize("failure", [ValueError, TurnCancelled, provider.RetryableStreamError])
def test_anthropic_mapping_is_lazy_and_preserves_separate_reasoning_blocks(monkeypatch, failure):
    first = [{"type": "thinking", "thinking": "first", "signature": "one"}]
    second = [{"type": "thinking", "thinking": "second", "signature": "two"}]
    events = [
        {"type": "text_delta", "delta": "answer"},
        {"type": "reasoning_done", "id": "r1", "provider_content": first},
        {"type": "tool_call_started", "id": "t1", "name": "read"},
        {"type": "reasoning_done", "id": "r2", "provider_content": second},
        {"type": "tool_call", "id": "t1", "name": "read", "arguments": '{"input":"a"}'},
    ]
    consumed = []
    error = failure("interrupted")

    def stream(*_args, **_kwargs):
        for event in events:
            consumed.append(event)
            yield event
        raise error

    monkeypatch.setattr(anthropic_http, "build_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(anthropic_http, "stream_message", stream)
    result = provider._stream_anthropic(None, {}, "model", "system", [], [])

    assert consumed == []
    assert next(result) == provider.TextDelta("answer")
    assert consumed == events[:1]
    assert next(result).provider_content is first
    assert next(result) == provider.ToolCallStarted(id="t1", call_id="t1", name="read")
    assert next(result).provider_content is second
    tool = next(result)
    assert tool.arguments == '{"input":"a"}'
    assert tool.provider_metadata == {"provider": "anthropic"}
    with pytest.raises(failure) as raised:
        next(result)
    assert raised.value is error


@pytest.mark.parametrize("winerror, attempts", [(2, 1), (5, 2)])
@pytest.mark.parametrize("operation", ["update", "uninstall"])
def test_windows_helper_launch_failure_preserves_retry_and_cleanup(
    monkeypatch, tmp_path, winerror, attempts, operation,
):
    staging = tmp_path / "staging"
    staging.mkdir()
    source = staging / "jarv.exe"
    source.write_bytes(b"candidate")
    target = tmp_path / "jarv.exe"
    target.write_bytes(b"installed")
    calls = []
    error = OSError("launch failed")
    error.winerror = winerror

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        raise error

    monkeypatch.setattr(standalone.subprocess, "Popen", popen)
    monkeypatch.setattr(standalone.shutil, "which", lambda _name: "powershell")
    monkeypatch.setattr(standalone, "WINDOWS_UPDATE_RESULT_FILE", tmp_path / "update-result.json")
    monkeypatch.setattr(uninstall, "UNINSTALL_RESULT_FILE", tmp_path / "uninstall-result.json")
    monkeypatch.setattr(uninstall.tempfile, "mkdtemp", lambda **_kwargs: str(staging))

    with pytest.raises(OSError) as raised:
        if operation == "update":
            standalone._stage_windows_updater(source, target, "1.0")
        else:
            uninstall._stage_windows_uninstaller(target, remove_path_entry=False, purge=False)

    assert raised.value is error
    assert len(calls) == attempts
    assert calls[0][1]["creationflags"] == standalone._windows_updater_creation_flags()
    if attempts == 2:
        assert calls[1][1]["creationflags"] == standalone._windows_updater_creation_flags(allow_breakaway=False)
        assert calls[1][0] == calls[0][0]
    assert target.read_bytes() == b"installed"
    assert staging.exists() is (operation == "update")
