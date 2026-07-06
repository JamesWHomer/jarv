"""Project context ingestion: AGENTS.md-style files and lightweight git awareness.

At the start of each agent run, jarv looks for a project instructions file
(JARV.md, AGENTS.md, or CLAUDE.md) in the working directory — walking up to
the git root when inside a repository — and builds a small git summary
(branch, clean/dirty state, recent commits). Both are appended to the system
prompt so the model starts each session aware of the project it is in.

Everything here fails silent: no context file, no git binary, a timeout, or
an unreadable file all degrade to omitting that block, never to an error.
"""

import os
import subprocess
from pathlib import Path

from .config_schema import get_setting
from .shell import truncate_model_output

CONTEXT_FILENAMES = ("JARV.md", "AGENTS.md", "CLAUDE.md")
GIT_TIMEOUT_SECONDS = 1.5
_MAX_WALK_DEPTH = 30

# Process-lifetime memo: once the git binary is found missing, skip every
# later spawn attempt instead of paying a failed process launch per turn.
_git_unavailable = False


def _run_git(args: list[str], cwd: Path) -> str | None:
    global _git_unavailable
    if _git_unavailable:
        return None
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        _git_unavailable = True
        return None
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_toplevel(cwd: Path) -> Path | None:
    out = _run_git(["rev-parse", "--show-toplevel"], cwd)
    return Path(out) if out else None


def find_context_file(cwd: Path, git_root: Path | None) -> Path | None:
    if git_root is None:
        candidates = [cwd]
    else:
        stop = os.path.normcase(str(git_root.resolve()))
        candidates = []
        for directory in [cwd, *cwd.parents][:_MAX_WALK_DEPTH]:
            candidates.append(directory)
            if os.path.normcase(str(directory.resolve())) == stop:
                break
    for directory in candidates:
        for name in CONTEXT_FILENAMES:
            path = directory / name
            if path.is_file():
                return path
    return None


def read_context_file(path: Path, max_chars: int) -> str:
    try:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""
    if not content.strip():
        return ""
    return truncate_model_output(content, max_chars, label="project context file")


def build_git_block(cwd: Path) -> str:
    lines = []
    branch = _run_git(["branch", "--show-current"], cwd)
    if branch:
        lines.append(f"branch: {branch}")
    elif branch is not None:
        sha = _run_git(["rev-parse", "--short", "HEAD"], cwd)
        if sha:
            lines.append(f"branch: detached HEAD at {sha}")
    status = _run_git(["status", "--porcelain"], cwd)
    if status is not None:
        changed = sum(1 for line in status.splitlines() if line.strip())
        if changed == 0:
            lines.append("status: clean")
        else:
            noun = "file" if changed == 1 else "files"
            lines.append(f"status: dirty ({changed} {noun} changed)")
    log = _run_git(["log", "--oneline", "-5", "--no-decorate"], cwd)
    if log:
        lines.append("recent commits:")
        lines.extend(f"  {line}" for line in log.splitlines())
    return "\n".join(lines)


def build_project_context(config: dict) -> str:
    if not get_setting(config, "project_context"):
        return ""
    cwd = Path(os.getcwd())
    git_root = _git_toplevel(cwd)

    parts = []
    context_path = find_context_file(cwd, git_root)
    if context_path is not None:
        content = read_context_file(
            context_path, get_setting(config, "project_context_max_chars")
        )
        if content:
            parts.append(
                f"Project context (from {context_path}) — project-provided instructions:\n{content}"
            )
    if git_root is not None:
        git_block = build_git_block(cwd)
        if git_block:
            parts.append(f"Git:\n{git_block}")
    return "\n\n".join(parts)
