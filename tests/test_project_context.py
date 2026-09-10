import shutil
import subprocess
import threading
from pathlib import Path

import pytest

import jarv.project_context as project_context
from jarv.project_context import (
    build_git_block,
    build_project_context,
    find_context_file,
    read_context_file,
)


@pytest.fixture(autouse=True)
def reset_git_memo():
    project_context._git_unavailable = False
    yield
    project_context._git_unavailable = False


def _fake_git(monkeypatch, responses, calls=None):
    """Patch _run_git with a canned args-tuple -> output map (None = failure)."""

    def fake(args, cwd):
        if calls is not None:
            calls.append(tuple(args))
        return responses.get(tuple(args))

    monkeypatch.setattr(project_context, "_run_git", fake)


def _no_repo(monkeypatch, calls=None):
    _fake_git(monkeypatch, {}, calls)


# --- file lookup ---


def test_finds_agents_md_in_cwd(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("Use tabs, not spaces.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _no_repo(monkeypatch)

    result = build_project_context({})

    assert "Use tabs, not spaces." in result
    assert "Project context (from " in result
    assert "AGENTS.md" in result


def test_priority_order_jarv_then_agents_then_claude(tmp_path):
    for name in ("JARV.md", "AGENTS.md", "CLAUDE.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    assert find_context_file(tmp_path, None).name == "JARV.md"

    (tmp_path / "JARV.md").unlink()
    assert find_context_file(tmp_path, None).name == "AGENTS.md"

    (tmp_path / "AGENTS.md").unlink()
    assert find_context_file(tmp_path, None).name == "CLAUDE.md"


def test_walkup_finds_file_in_parent_inside_repo(tmp_path):
    root = tmp_path / "repo"
    cwd = root / "src" / "pkg"
    cwd.mkdir(parents=True)
    (root / "AGENTS.md").write_text("root instructions", encoding="utf-8")

    found = find_context_file(cwd, git_root=root)

    assert found == root / "AGENTS.md"


def test_walkup_stops_at_git_root(tmp_path):
    (tmp_path / "AGENTS.md").write_text("above the repo", encoding="utf-8")
    root = tmp_path / "repo"
    cwd = root / "src"
    cwd.mkdir(parents=True)

    assert find_context_file(cwd, git_root=root) is None


def test_no_walkup_outside_repo(tmp_path):
    (tmp_path / "AGENTS.md").write_text("parent file", encoding="utf-8")
    cwd = tmp_path / "sub"
    cwd.mkdir()

    assert find_context_file(cwd, git_root=None) is None
    (cwd / "AGENTS.md").write_text("cwd file", encoding="utf-8")
    assert find_context_file(cwd, git_root=None) == cwd / "AGENTS.md"


def test_directory_named_like_context_file_is_ignored(tmp_path):
    (tmp_path / "AGENTS.md").mkdir()

    assert find_context_file(tmp_path, None) is None


def test_non_repo_has_no_git_block(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("instructions", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _no_repo(monkeypatch)

    result = build_project_context({})

    assert "instructions" in result
    assert "Git:" not in result


# --- content handling ---


def test_empty_or_whitespace_file_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("  \n\t\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _no_repo(monkeypatch)

    assert build_project_context({}) == ""


def test_oversized_file_is_truncated(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("x" * 5_000, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _no_repo(monkeypatch)

    result = build_project_context({"project_context_max_chars": 500})

    assert len(result) < 1_500
    assert "truncated" in result


def test_bom_is_stripped(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("Hello", encoding="utf-8-sig")
    monkeypatch.chdir(tmp_path)
    _no_repo(monkeypatch)

    result = build_project_context({})

    assert "﻿" not in result
    assert "Hello" in result


def test_invalid_utf8_does_not_raise(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_bytes(b"valid \x80\xff invalid")
    monkeypatch.chdir(tmp_path)
    _no_repo(monkeypatch)

    result = build_project_context({})

    assert "valid" in result


def test_unreadable_file_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("secret", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _no_repo(monkeypatch)

    def raise_oserror(self, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", raise_oserror)

    assert build_project_context({}) == ""


# --- git block ---


def _repo_responses(root):
    return {
        ("rev-parse", "--show-toplevel"): str(root),
        ("branch", "--show-current"): "main",
        ("status", "--porcelain"): " M a.py\n?? b.py\n M c.py",
        ("log", "--oneline", "-5", "--no-decorate"): "abc1 one\nabc2 two\nabc3 three\nabc4 four\nabc5 five",
    }


def test_git_block_formatting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_git(monkeypatch, _repo_responses(tmp_path))

    result = build_project_context({})

    assert "Git:\nbranch: main\nstatus: dirty (3 files changed)\nrecent commits:" in result
    assert "  abc1 one" in result
    assert "  abc5 five" in result


def test_git_block_clean_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_git(
        monkeypatch,
        {**_repo_responses(tmp_path), ("status", "--porcelain"): ""},
    )

    assert "status: clean" in build_project_context({})


def test_detached_head_uses_short_sha(monkeypatch, tmp_path):
    _fake_git(
        monkeypatch,
        {
            ("branch", "--show-current"): "",
            ("rev-parse", "--short", "HEAD"): "abc1234",
            ("status", "--porcelain"): "",
        },
    )

    block = build_git_block(tmp_path)

    assert "branch: detached HEAD at abc1234" in block


def test_git_failures_omit_lines(monkeypatch, tmp_path):
    _fake_git(monkeypatch, {("status", "--porcelain"): ""})

    assert build_git_block(tmp_path) == "status: clean"
    _fake_git(monkeypatch, {})
    assert build_git_block(tmp_path) == ""


def test_git_queries_overlap_without_reordering_output(monkeypatch, tmp_path):
    started = threading.Barrier(3, timeout=5)
    responses = _repo_responses(tmp_path)

    def query(args, cwd):
        # All three must start before any can finish; no wall-clock threshold.
        started.wait()
        return responses[tuple(args)]

    monkeypatch.setattr(project_context, "_run_git", query)
    assert build_git_block(tmp_path) == (
        "branch: main\nstatus: dirty (3 files changed)\nrecent commits:\n"
        "  abc1 one\n  abc2 two\n  abc3 three\n  abc4 four\n  abc5 five"
    )


def test_git_status_timeout_preserves_other_context(monkeypatch, tmp_path):
    def run(command, **kwargs):
        if command[1] == "status":
            raise subprocess.TimeoutExpired(command, timeout=1.5)
        output = "main" if command[1] == "branch" else "abc123 initial"
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(project_context.subprocess, "run", run)
    assert build_git_block(tmp_path) == "branch: main\nrecent commits:\n  abc123 initial"


@pytest.mark.parametrize("state", ["unborn", "clean", "dirty", "detached"])
def test_git_context_real_repository(tmp_path, state):
    if not shutil.which("git"):
        pytest.skip("git is not installed")

    def git(*args):
        result = subprocess.run(
            ["git", "-c", "user.name=Benchmark Test", "-c", "user.email=test@example.invalid",
             "-c", "commit.gpgsign=false", *args], cwd=tmp_path,
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()

    git("init", "-b", "main")
    if state == "unborn":
        assert build_git_block(tmp_path) == "branch: main\nstatus: clean"
        return
    (tmp_path / "tracked.txt").write_text("original\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "initial")
    commit = git("log", "--oneline", "-1", "--no-decorate")
    branch = "main"
    status = "clean"
    if state == "dirty":
        (tmp_path / "tracked.txt").write_text("modified\n", encoding="utf-8")
        (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")
        status = "dirty (2 files changed)"
    elif state == "detached":
        git("checkout", "--detach")
        branch = f"detached HEAD at {git('rev-parse', '--short', 'HEAD')}"
    assert build_git_block(tmp_path) == f"branch: {branch}\nstatus: {status}\nrecent commits:\n  {commit}"


def test_missing_git_sets_memo_and_stops_spawning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    attempts = []

    def raise_not_found(*args, **kwargs):
        attempts.append(args)
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(project_context.subprocess, "run", raise_not_found)

    assert build_project_context({}) == ""
    assert project_context._git_unavailable is True
    assert len(attempts) == 1

    assert build_project_context({}) == ""
    assert len(attempts) == 1


def test_git_timeout_is_silent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1.5)

    monkeypatch.setattr(project_context.subprocess, "run", raise_timeout)

    assert build_project_context({}) == ""
    assert project_context._git_unavailable is False


# --- toggle ---


def test_disabled_project_context_skips_everything(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("instructions", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = []
    _no_repo(monkeypatch, calls)

    assert build_project_context({"project_context": False}) == ""
    assert calls == []


# --- instructions wiring ---


def test_build_instructions_appends_project_context(monkeypatch):
    from jarv import agent

    monkeypatch.setattr(agent, "get_system_info", lambda cwd=None: "SI")
    monkeypatch.setattr(agent, "build_project_context", lambda config, cwd=None: "PC")

    assert agent.build_instructions({"system_prompt": "SP"}) == "SP\n\nSystem info:\nSI\n\nPC"


def test_build_instructions_without_project_context(monkeypatch):
    from jarv import agent

    monkeypatch.setattr(agent, "get_system_info", lambda cwd=None: "SI")
    monkeypatch.setattr(agent, "build_project_context", lambda config, cwd=None: "")

    assert agent.build_instructions({"system_prompt": "SP"}) == "SP\n\nSystem info:\nSI"
