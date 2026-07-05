import codecs
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
    target.write_text("alpha\n", encoding="utf-8")

    def _fail(self, data):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "write_bytes", _fail)

    output = _edit(_args(target))

    assert output.startswith("[edit error: could not write file:")


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
