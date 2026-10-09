import codecs
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from jarv import edit_tool
from jarv.config import DEFAULT_CONFIG
from jarv.edit_tool import (
    EDIT_TOOL,
    classify_edit,
    dispatch_edit_tool,
)
from jarv.safety import prompt_confirmation


NO_PROMPT_CONFIG = {**DEFAULT_CONFIG, "command_safety": "none"}


def test_replace_all_rejects_growth_before_allocating_replacement():
    class UnallocatedReplacement(str):
        def replace(self, *args, **kwargs):
            raise AssertionError("oversized replacement must not be allocated")

    result = edit_tool._apply_replacement(
        UnallocatedReplacement("a" * 100_000), "a", "b" * 100, True,
        path=Path("file.txt"),
    )
    assert "would produce 10000000 bytes" in result


def test_edit_growth_rejection_keeps_file_and_skips_diff(workdir, monkeypatch):
    target = workdir / "file.txt"
    original = b"a" * 100_000
    target.write_bytes(original)

    def unexpected_diff(*args, **kwargs):
        raise AssertionError("rejected edit must not build a diff or request approval")

    monkeypatch.setattr(edit_tool, "build_edit_diff", unexpected_diff)
    result = _edit(_args(target, old="a", new="b" * 100, replace_all=True))
    assert "would produce 10000000 bytes" in result
    assert target.read_bytes() == original
    assert not list(workdir.glob(".jarv-edit-*"))


@pytest.mark.parametrize("original,old,new,replace_all,expected", [
    (b"a", "a", "é", False, "é".encode()),
    (codecs.BOM_UTF8 + b"a", "a", "é", False, codecs.BOM_UTF8 + "é".encode()),
    (b"a\r\nb", "a\nb", "é\n中", False, "é\r\n中".encode()),
    (b"aaaa", "a", "é", True, "éééé".encode()),
])
@pytest.mark.parametrize("below_limit", [False, True])
def test_edit_growth_counts_encoded_unicode_bom_and_crlf(
    workdir, monkeypatch, original, old, new, replace_all, expected, below_limit,
):
    target = workdir / "file.txt"
    target.write_bytes(original)
    monkeypatch.setattr(edit_tool, "MAX_EDIT_FILE_BYTES", len(expected) - below_limit)
    result = _edit(_args(target, old=old, new=new, replace_all=replace_all))
    if below_limit:
        assert f"would produce {len(expected)} bytes" in result
        assert target.read_bytes() == original
    else:
        assert result.startswith("[EDIT RESULT]")
        assert target.read_bytes() == expected


def test_edit_invalid_unicode_replacement_is_a_failed_result(workdir):
    target = workdir / "file.txt"
    target.write_bytes(b"alpha")
    result = _edit(_args(target, new="\ud800"))
    assert "must be valid UTF-8" in result
    assert target.read_bytes() == b"alpha"


def test_edit_minified_file_previews_have_independent_character_limits(workdir):
    target = workdir / "file.txt"
    original = "alpha" + "x" * 100_000
    expected = "beta" + "x" * 100_000
    target.write_text(original, encoding="utf-8")
    diff = edit_tool.build_edit_diff(original, expected, str(target))
    result = _edit(_args(target))
    assert len(diff) <= edit_tool._MAX_DIFF_PREVIEW_CHARS
    assert len(result) <= edit_tool._MAX_RESULT_PREVIEW_CHARS
    assert "preview truncated" in diff
    assert "preview truncated" in result
    assert target.read_text(encoding="utf-8") == expected


def _edit(args, config=None):
    return dispatch_edit_tool(args, config=config or NO_PROMPT_CONFIG)


def _args(path, old="alpha", new="beta", **extra):
    return {"path": str(path), "old_text": old, "new_text": new, **extra}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ── Schema ────────────────────────────────────────────────────────────────

def test_edit_schema_shape():
    assert EDIT_TOOL["name"] == "edit"
    parameters = EDIT_TOOL["parameters"]
    assert parameters["required"] == ["path", "old_text", "new_text"]
    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["replace_all"]["type"] == "boolean"
    assert parameters["properties"]["old_text"]["minLength"] == 1


# ── Argument validation ───────────────────────────────────────────────────

def test_edit_rejects_non_dict_args():
    assert dispatch_edit_tool("nope", config=NO_PROMPT_CONFIG) == (
        "[tool argument error: edit arguments must be an object]"
    )


@pytest.mark.parametrize(
    "args, expected_fragment",
    [
        ({"old_text": "a", "new_text": "b"}, "path must be a non-empty string"),
        ({"path": "  ", "old_text": "a", "new_text": "b"}, "path must be a non-empty string"),
        ({"path": "f", "old_text": "", "new_text": "b"}, "old_text must be a non-empty string"),
        ({"path": "f", "old_text": 5, "new_text": "b"}, "old_text must be a non-empty string"),
        ({"path": "f", "old_text": "a"}, "new_text must be a string"),
        ({"path": "f", "old_text": "a", "new_text": "a"}, "identical"),
        ({"path": "f", "old_text": "a", "new_text": "b", "replace_all": "yes"}, "replace_all must be a boolean"),
    ],
)
def test_edit_argument_errors(args, expected_fragment):
    output = _edit(args)
    assert output.startswith("[tool argument error:")
    assert expected_fragment in output


def test_edit_replace_all_null_defaults_to_false(workdir):
    target = workdir / "file.txt"
    target.write_text("alpha\n", encoding="utf-8")

    output = _edit(_args(target, replace_all=None))

    assert output.startswith("[EDIT RESULT]")
    assert target.read_text(encoding="utf-8") == "beta\n"


def test_edit_crlf_fallback_accepts_crlf_replacement(workdir):
    target = workdir / "file.txt"
    target.write_bytes(b"alpha\r\nbeta\r\n")
    output = _edit(_args(target, old="alpha\nbeta", new="gamma\r\ndelta"))
    assert output.startswith("[EDIT RESULT]")
    assert target.read_bytes() == b"gamma\r\ndelta\r\n"


def test_edit_invalid_path_is_a_failed_tool_result():
    from jarv.tool_outputs import tool_outcome

    output = _edit(_args("bad\x00path"))
    assert output.startswith("[edit error:")
    assert tool_outcome(output).status == "failed"


def test_edit_unresolvable_home_returns_tool_error(monkeypatch):
    def missing_home(path):
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "expanduser", missing_home)
    assert _edit(_args("~/file.txt")).startswith("[edit error:")


# ── Path resolution ───────────────────────────────────────────────────────

def test_edit_missing_file_errors_and_does_not_create(workdir):
    target = workdir / "missing.txt"

    output = _edit(_args(target))

    assert output.startswith("[edit error: file not found:")
    assert "use run_command to create files" in output
    assert not target.exists()


def test_edit_rejects_directory(workdir):
    output = _edit(_args(workdir))
    assert output.startswith("[edit error: path is not a file:")


def test_edit_resolves_relative_path_from_cwd(workdir):
    (workdir / "rel.txt").write_text("alpha\n", encoding="utf-8")

    output = _edit(_args("rel.txt"))

    assert output.startswith("[EDIT RESULT]")
    assert (workdir / "rel.txt").read_text(encoding="utf-8") == "beta\n"


# ── Match / replace core ──────────────────────────────────────────────────

def test_edit_unique_match_reports_result_block(workdir):
    target = workdir / "code.py"
    target.write_text("def foo():\n    return 1\n\nprint(foo())\n", encoding="utf-8")

    output = _edit(_args(target, old="    return 1", new="    return 2"))

    assert output.startswith("[EDIT RESULT]")
    assert f"Path: {target.resolve()}" in output
    assert "Replacements: 1" in output
    assert "Lines: 4 -> 4 (+0)" in output
    assert "Context (new file content around first change):" in output
    assert "2 |     return 2" in output
    assert target.read_text(encoding="utf-8") == "def foo():\n    return 2\n\nprint(foo())\n"


def test_edit_zero_matches_leaves_file_untouched(workdir):
    target = workdir / "file.txt"
    target.write_text("alpha\n", encoding="utf-8")

    output = _edit(_args(target, old="ALPHA"))

    assert output.startswith("[edit error: old_text not found")
    assert "copy old_text exactly" in output
    assert target.read_text(encoding="utf-8") == "alpha\n"


def test_edit_ambiguous_match_requires_replace_all(workdir):
    target = workdir / "file.txt"
    target.write_text("alpha\nalpha\n", encoding="utf-8")

    output = _edit(_args(target))

    assert "matches 2 locations" in output
    assert "replace_all=true" in output
    assert target.read_text(encoding="utf-8") == "alpha\nalpha\n"


def test_edit_replace_all_reports_count(workdir):
    target = workdir / "file.txt"
    target.write_text("alpha one alpha two alpha\n", encoding="utf-8")

    output = _edit(_args(target, replace_all=True))

    assert "Replacements: 3" in output
    assert target.read_text(encoding="utf-8") == "beta one beta two beta\n"


def test_edit_replace_all_uses_non_overlapping_matches(workdir):
    target = workdir / "file.txt"
    target.write_text("aaa\n", encoding="utf-8")

    output = _edit(_args(target, old="aa", new="b", replace_all=True))

    assert "Replacements: 1" in output
    assert target.read_text(encoding="utf-8") == "ba\n"


def test_edit_empty_new_text_deletes(workdir):
    target = workdir / "file.txt"
    target.write_text("keep alpha keep\n", encoding="utf-8")

    output = _edit(_args(target, old=" alpha", new=""))

    assert output.startswith("[EDIT RESULT]")
    assert target.read_text(encoding="utf-8") == "keep keep\n"


# ── Encoding and line endings ─────────────────────────────────────────────

def test_edit_crlf_fallback_preserves_crlf(workdir):
    target = workdir / "file.txt"
    target.write_bytes(b"one\r\ntwo\r\nthree\r\n")

    output = _edit(_args(target, old="one\ntwo", new="uno\ntwo"))

    assert output.startswith("[EDIT RESULT]")
    assert target.read_bytes() == b"uno\r\ntwo\r\nthree\r\n"


def test_edit_lf_file_stays_lf_and_keeps_trailing_newline(workdir):
    target = workdir / "file.txt"
    target.write_bytes(b"one\ntwo\n")

    _edit(_args(target, old="two", new="dos"))

    assert target.read_bytes() == b"one\ndos\n"


def test_edit_preserves_missing_trailing_newline(workdir):
    target = workdir / "file.txt"
    target.write_bytes(b"one\ntwo")

    _edit(_args(target, old="two", new="dos"))

    assert target.read_bytes() == b"one\ndos"


def test_edit_preserves_utf8_bom(workdir):
    target = workdir / "file.txt"
    target.write_bytes(codecs.BOM_UTF8 + b"alpha\n")

    output = _edit(_args(target))

    assert output.startswith("[EDIT RESULT]")
    assert target.read_bytes() == codecs.BOM_UTF8 + b"beta\n"


def test_edit_rejects_binary_file(workdir):
    target = workdir / "blob.bin"
    target.write_bytes(b"al\x00pha")

    assert "binary file" in _edit(_args(target))


def test_edit_rejects_invalid_utf8(workdir):
    target = workdir / "latin.txt"
    target.write_bytes(b"caf\xe9 alpha")

    assert "not valid UTF-8" in _edit(_args(target))


def test_edit_rejects_oversized_file(workdir, monkeypatch):
    monkeypatch.setattr(edit_tool, "MAX_EDIT_FILE_BYTES", 4)
    target = workdir / "big.txt"
    target.write_text("alpha\n", encoding="utf-8")

    output = _edit(_args(target))

    assert "byte edit limit" in output
    assert "run_command" in output


# ── Risk classification ───────────────────────────────────────────────────

def test_classify_edit_plain_file_in_cwd_is_not_risky(workdir):
    target = workdir / "src" / "main.py"
    assert classify_edit(target) == (False, "")


def test_classify_edit_outside_cwd(workdir):
    outside = workdir.parent / "elsewhere.txt"
    risky, reason = classify_edit(outside)
    assert risky
    assert reason == "file outside the current working directory"


@pytest.mark.parametrize(
    "name", [".env", ".env.local", "server.pem", "signing.key", "id_rsa", ".npmrc"]
)
def test_classify_edit_sensitive_files(workdir, name):
    risky, reason = classify_edit(workdir / name)
    assert risky
    assert reason == "sensitive file (secrets/keys)"


def test_classify_edit_credentials_directory(workdir):
    risky, reason = classify_edit(workdir / ".ssh" / "config")
    assert risky
    assert reason == "credentials directory"


def test_classify_edit_hidden_file(workdir):
    risky, reason = classify_edit(workdir / ".git" / "config")
    assert risky
    assert reason == "hidden file or directory"


def test_classify_edit_system_path(workdir):
    if Path("C:/").exists():
        target = Path("C:/Windows/System32/drivers/etc/hosts")
    else:
        target = Path("/etc/hosts")
    risky, reason = classify_edit(target)
    assert risky
    assert reason == "system path"


def test_classify_edit_allows_dot_cwd(tmp_path, monkeypatch):
    project = tmp_path / ".config" / "project"
    project.mkdir(parents=True)
    monkeypatch.chdir(project)

    assert classify_edit(project.resolve() / "settings.toml") == (False, "")


# ── Safety gating ─────────────────────────────────────────────────────────

def test_edit_denied_under_safety_all(workdir, monkeypatch):
    target = workdir / "file.txt"
    target.write_text("alpha\n", encoding="utf-8")
    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", lambda *a, **k: False)

    output = _edit(_args(target), config={**DEFAULT_CONFIG, "command_safety": "all"})

    assert output == "[edit denied by user — all edits require approval]"
    assert target.read_text(encoding="utf-8") == "alpha\n"


def test_edit_approved_under_safety_all(workdir, monkeypatch):
    target = workdir / "file.txt"
    target.write_text("alpha\n", encoding="utf-8")
    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", lambda *a, **k: True)

    output = _edit(_args(target), config={**DEFAULT_CONFIG, "command_safety": "all"})

    assert output.startswith("[EDIT RESULT]")
    assert target.read_text(encoding="utf-8") == "beta\n"


def test_edit_risky_level_skips_prompt_for_clean_path(workdir, monkeypatch):
    target = workdir / "file.txt"
    target.write_text("alpha\n", encoding="utf-8")

    def _no_prompt(*args, **kwargs):
        raise AssertionError("prompt should not fire for a non-risky edit")

    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", _no_prompt)

    output = _edit(_args(target), config={**DEFAULT_CONFIG, "command_safety": "risky"})

    assert output.startswith("[EDIT RESULT]")


def test_edit_risky_level_prompts_for_sensitive_file(workdir, monkeypatch):
    target = workdir / ".env"
    target.write_text("alpha\n", encoding="utf-8")
    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", lambda *a, **k: False)

    output = _edit(_args(target), config={**DEFAULT_CONFIG, "command_safety": "risky"})

    assert output == "[edit denied by user — sensitive file (secrets/keys)]"
    assert target.read_text(encoding="utf-8") == "alpha\n"


def test_edit_safety_none_never_prompts(workdir, monkeypatch):
    target = workdir / ".env"
    target.write_text("alpha\n", encoding="utf-8")

    def _no_prompt(*args, **kwargs):
        raise AssertionError("prompt should not fire when command_safety is none")

    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", _no_prompt)

    output = _edit(_args(target))

    assert output.startswith("[EDIT RESULT]")


# ── Write failures ────────────────────────────────────────────────────────

def test_edit_reports_write_failure(workdir, monkeypatch):
    target = workdir / "file.txt"
    target.write_bytes(b"alpha\n")

    def _fail(source, destination):
        raise PermissionError("locked")

    monkeypatch.setattr(edit_tool.os, "replace", _fail)

    output = _edit(_args(target))

    assert output.startswith("[edit error: could not write file:")
    assert target.read_bytes() == b"alpha\n"
    assert list(workdir.iterdir()) == [target]


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only file semantics")
@pytest.mark.parametrize("cancel", [False, True])
def test_read_only_edit_cleans_staging_file_and_preserves_failure(workdir, monkeypatch, cancel):
    from jarv.cancellation import CancellationToken, TurnCancelled

    target = workdir / "file.txt"
    target.write_bytes(b"alpha\n")
    target.chmod(stat.S_IREAD)
    token = CancellationToken()
    if cancel:
        monkeypatch.setattr(edit_tool.os, "fsync", lambda fd: token.cancel())
    try:
        if cancel:
            with pytest.raises(TurnCancelled):
                dispatch_edit_tool(_args(target), config=NO_PROMPT_CONFIG, cancellation_token=token)
        else:
            output = _edit(_args(target))
            assert output.startswith("[edit error: could not write file:")
        assert target.read_bytes() == b"alpha\n"
        assert not target.stat().st_mode & stat.S_IWRITE
        assert list(workdir.iterdir()) == [target]
    finally:
        for path in workdir.iterdir():
            path.chmod(stat.S_IWRITE)


@pytest.mark.parametrize("change", ["write", "delete", "replace", "same_stat"])
def test_edit_rejects_changes_during_approval(workdir, monkeypatch, change):
    target = workdir / "file.txt"
    target.write_bytes(b"alpha\n")
    original = target.stat()

    def approve(*args, **kwargs):
        if change == "delete":
            target.unlink()
        elif change == "replace":
            replacement = workdir / "replacement"
            replacement.write_bytes(b"alpha\n")
            os.replace(replacement, target)
        else:
            target.write_bytes(b"omega\n")
            if change == "same_stat":
                os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        return True

    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", approve)
    output = _edit(_args(target), config={**DEFAULT_CONFIG, "command_safety": "all"})

    assert output.startswith("[edit conflict:")
    assert "Read the file and retry" in output
    if change == "delete":
        assert not target.exists()
    else:
        assert target.read_bytes() == (b"alpha\n" if change == "replace" else b"omega\n")
    assert not list(workdir.glob(".jarv-edit-*"))


def test_edit_serializes_same_file_but_allows_other_files(workdir, monkeypatch):
    target = workdir / "file.txt"
    target.write_bytes(b"alpha one\n")
    other = workdir / "other.txt"
    other.write_bytes(b"alpha\n")
    awaiting_approval = threading.Event()
    release = threading.Event()
    second_waiting = threading.Event()
    real_lock = edit_tool._file_edit_lock

    def observe_lock(path, token):
        if threading.current_thread().name.endswith("_1"):
            second_waiting.set()
        return real_lock(path, token)

    def approve(path, diff, config, **kwargs):
        if path == target.resolve() and "beta" in diff:
            awaiting_approval.set()
            assert release.wait(5)
        return True, ""

    monkeypatch.setattr(edit_tool, "_check_edit", approve)
    monkeypatch.setattr(edit_tool, "_file_edit_lock", observe_lock)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(_edit, _args(target))
        try:
            assert awaiting_approval.wait(5)
            second = pool.submit(_edit, _args("file.txt", old="one", new="two"))
            assert second_waiting.wait(5)
            assert not second.done()
            independent = pool.submit(_edit, _args(other))
            assert independent.result(timeout=5).startswith("[EDIT RESULT]")
        finally:
            release.set()
        assert first.result(timeout=5).startswith("[EDIT RESULT]")
        assert second.result(timeout=5).startswith("[EDIT RESULT]")
    assert target.read_bytes() == b"beta two\n"


def test_edit_stages_complete_file_before_atomic_replace(workdir, monkeypatch):
    target = workdir / "file.txt"
    target.write_bytes(codecs.BOM_UTF8 + b"alpha\r\n")
    real_replace = os.replace
    seen = []

    def replace(source, destination):
        assert Path(source).parent == target.parent
        assert target.read_bytes() == codecs.BOM_UTF8 + b"alpha\r\n"
        assert Path(source).read_bytes() == codecs.BOM_UTF8 + b"beta\r\n"
        assert Path(source).stat().st_mode == target.stat().st_mode
        seen.append(True)
        real_replace(source, destination)

    monkeypatch.setattr(edit_tool.os, "replace", replace)
    assert _edit(_args(target)).startswith("[EDIT RESULT]")
    assert seen == [True]
    assert list(workdir.iterdir()) == [target]


def test_edit_staging_failure_preserves_original(workdir, monkeypatch):
    target = workdir / "file.txt"
    target.write_bytes(b"alpha\n")

    def fail(fd):
        raise OSError("disk full")

    monkeypatch.setattr(edit_tool.os, "fsync", fail)
    assert _edit(_args(target)).startswith("[edit error: could not write file:")
    assert target.read_bytes() == b"alpha\n"
    assert list(workdir.iterdir()) == [target]


def test_edit_cancelled_while_waiting_for_file_lock(workdir):
    from jarv.cancellation import CancellationToken, TurnCancelled

    target = workdir / "file.txt"
    target.write_bytes(b"alpha\n")
    token = CancellationToken()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with edit_tool._file_edit_lock(target.resolve(), None):
            pending = pool.submit(dispatch_edit_tool, _args(target),
                                  config=NO_PROMPT_CONFIG, cancellation_token=token)
            token.cancel()
            with pytest.raises(TurnCancelled):
                pending.result(timeout=5)
    assert target.read_bytes() == b"alpha\n"
    assert _edit(_args(target)).startswith("[EDIT RESULT]")


def test_edit_cancelled_after_staging_cleans_up(workdir, monkeypatch):
    from jarv.cancellation import CancellationToken, TurnCancelled

    target = workdir / "file.txt"
    target.write_bytes(b"alpha\n")
    token = CancellationToken()
    monkeypatch.setattr(edit_tool.os, "fsync", lambda fd: token.cancel())
    with pytest.raises(TurnCancelled):
        dispatch_edit_tool(_args(target), config=NO_PROMPT_CONFIG, cancellation_token=token)
    assert target.read_bytes() == b"alpha\n"
    assert list(workdir.iterdir()) == [target]


def test_edit_approval_lock_wait_is_cancellable(workdir, monkeypatch):
    from jarv.cancellation import CancellationToken, TurnCancelled

    target = workdir / "file.txt"
    target.write_bytes(b"alpha\n")
    token = CancellationToken()
    waiting = threading.Event()
    held_lock = threading.Lock()

    class ObservedLock:
        def acquire(self, timeout):
            waiting.set()
            return held_lock.acquire(timeout=timeout)

        def release(self):
            held_lock.release()

    monkeypatch.setattr(edit_tool, "approval_lock", lambda: ObservedLock())
    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", lambda *a, **k: pytest.fail("cancelled edit prompted"))
    config = {**DEFAULT_CONFIG, "command_safety": "all"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        with held_lock:
            pending = pool.submit(dispatch_edit_tool, _args(target), config=config, cancellation_token=token)
            assert waiting.wait(1)
            token.cancel()
            with pytest.raises(TurnCancelled):
                pending.result(timeout=1)
    assert target.read_bytes() == b"alpha\n"
    assert held_lock.acquire(blocking=False)
    held_lock.release()


@pytest.mark.parametrize("explicit_child", [False, True])
@pytest.mark.parametrize("approved", [False, True])
def test_edit_approval_uses_token_without_deadline_and_checks_after_prompt(
    workdir, monkeypatch, explicit_child, approved,
):
    from types import SimpleNamespace
    from jarv.cancellation import CancellationToken, TurnCancelled

    target = workdir / "file.txt"
    target.write_bytes(b"alpha\n")
    parent = CancellationToken()
    child = CancellationToken() if explicit_child else None
    expected = child or parent
    config = {**DEFAULT_CONFIG, "command_safety": "all",
              "_run_control": SimpleNamespace(token=parent, deadline=None)}

    def prompt(*args, **kwargs):
        assert kwargs["cancellation_token"] is expected
        expected.cancel()
        return approved

    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", prompt)
    with pytest.raises(TurnCancelled):
        dispatch_edit_tool(_args(target), config=config, cancellation_token=child)
    assert target.read_bytes() == b"alpha\n"
    assert edit_tool.approval_lock().acquire(blocking=False)
    edit_tool.approval_lock().release()
    if explicit_child:
        assert not parent.cancelled


@pytest.mark.parametrize("control", ["\r", "\x85", "\x0b", "\x0c"])
def test_edit_diff_keeps_controls_visible_when_replacing_with_newline(workdir, monkeypatch, control):
    import io
    from rich.console import Console
    from jarv.terminal_text import safe_terminal_text

    target = workdir / "file.txt"
    original = f"before{control}after"
    target.write_bytes(original.encode("utf-8"))
    captured = io.StringIO()

    def deny(body, **kwargs):
        Console(file=captured, color_system=None, width=120).print(body)
        return False

    monkeypatch.setattr(edit_tool, "prompt_panel_confirmation", deny)
    output = dispatch_edit_tool(
        _args(target, old=original, new="before\nafter"),
        config={**DEFAULT_CONFIG, "command_safety": "all"},
    )
    shown = captured.getvalue()
    assert "[edit denied" in output
    assert f"-before{safe_terminal_text(control)}after" in shown
    assert "+before" in shown and "+after" in shown
    assert target.read_bytes() == original.encode("utf-8")


def test_edit_diff_escapes_standalone_trailing_cr_before_adding_row_separator():
    diff = edit_tool.build_edit_diff("before\r", "after", "file.txt")
    assert "-before\\r\n+after" in diff
    # Actual CR and the literal two-character escape remain distinct for the
    # comparison even though their visible representations are identical.
    assert edit_tool.build_edit_diff("before\r", "before\\r", "file.txt")


def test_edit_diff_preserves_regular_crlf_and_empty_file_behavior():
    assert edit_tool.build_edit_diff("before\r\n", "after\r\n", "file.txt") == (
        "--- file.txt\n+++ file.txt\n@@ -1 +1 @@\n-before\n+after"
    )
    assert "@@ -0,0 +1 @@" in edit_tool.build_edit_diff("", "after\n", "file.txt")


# ── Orchestrator integration ──────────────────────────────────────────────

def _root_node():
    from jarv.orchestrator import AgentNode

    return AgentNode(label="root", depth=0, parent_label=None, task="", sterile=False)


def test_dispatch_tool_routes_edit(workdir):
    from jarv.artifacts import ArtifactStore
    from jarv.orchestrator import PARALLEL_SAFE_TOOL_NAMES, dispatch_tool

    target = workdir / "file.txt"
    target.write_text("alpha\n", encoding="utf-8")

    output = dispatch_tool(
        "edit", _args(target), _root_node(), ArtifactStore(), None, NO_PROMPT_CONFIG
    )

    assert output.startswith("[EDIT RESULT]")
    assert "edit" not in PARALLEL_SAFE_TOOL_NAMES


def test_dispatch_tool_respects_disabled_edit(workdir):
    from jarv.artifacts import ArtifactStore
    from jarv.orchestrator import dispatch_tool

    target = workdir / "file.txt"
    target.write_text("alpha\n", encoding="utf-8")
    config = {**NO_PROMPT_CONFIG, "disabled_tools": ["edit"]}

    output = dispatch_tool(
        "edit", _args(target), _root_node(), ArtifactStore(), None, config
    )

    assert output == "[tool disabled: edit]"
    assert target.read_text(encoding="utf-8") == "alpha\n"


# ── Safety refactor regression ────────────────────────────────────────────

@pytest.mark.parametrize("answer, expected", [("y", True), ("n", False)])
def test_prompt_confirmation_still_prompts_after_refactor(monkeypatch, answer, expected):
    from jarv import safety

    monkeypatch.setattr(safety.console, "input", lambda *a, **k: answer)
    monkeypatch.setattr(safety.console, "print", lambda *a, **k: None)

    assert prompt_confirmation("rm -rf build", "recursive deletion") is expected


@pytest.mark.parametrize("approve, expected_start", [(True, "[EDIT RESULT]"), (False, "[edit denied")])
def test_edit_confirmation_routes_through_confirm_handler(workdir, approve, expected_start):
    from jarv.safety import clear_confirm_handler, set_confirm_handler

    target = workdir / "file.txt"
    target.write_text("alpha\n", encoding="utf-8")
    config = {**DEFAULT_CONFIG, "command_safety": "all"}
    seen = {}

    def handler(request):
        seen["kind"] = request.kind
        seen["question"] = request.question
        return approve

    set_confirm_handler(handler)
    try:
        output = _edit(_args(target), config=config)
    finally:
        clear_confirm_handler()

    assert output.startswith(expected_start)
    assert seen["kind"] == "edit"
    assert seen["question"] == "Allow this edit?"
    assert target.read_text(encoding="utf-8") == ("beta\n" if approve else "alpha\n")
