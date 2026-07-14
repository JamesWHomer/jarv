"""Exact string-replacement editing of local text files."""

from __future__ import annotations

import codecs
import difflib
import os
from dataclasses import dataclass
from pathlib import Path

from rich.console import Group
from rich.markup import escape
from rich.text import Text

from .cancellation import CancellationToken
from .config import get_setting
from .safety import approval_lock, prompt_panel_confirmation
from .tool_outputs import ToolOutput


MAX_EDIT_FILE_BYTES = 5_000_000
_DIFF_CONTEXT_LINES = 3
_MAX_DIFF_PREVIEW_LINES = 60
_RESULT_CONTEXT_LINES = 3

EDIT_TOOL = {
    "type": "function",
    "name": "edit",
    "description": (
        "Make an exact string replacement in an existing UTF-8 text file. "
        "old_text must be copied verbatim from the file — including whitespace, "
        "indentation, and line breaks — and must match exactly one location "
        "unless replace_all is true. If the match is ambiguous, the edit fails; "
        "include more surrounding lines to make old_text unique, or set "
        "replace_all=true to change every occurrence. Read the file first so "
        "old_text is exact. Cannot create files; use run_command to create files. "
        "Depending on settings, the user may be asked to approve the edit."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to an existing file. Relative paths resolve from "
                    "the current working directory."
                ),
            },
            "old_text": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Exact existing text to replace, copied verbatim from the "
                    "file including indentation and line breaks."
                ),
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text. Empty string deletes old_text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": (
                    "Replace every occurrence of old_text. Defaults to false, "
                    "which requires exactly one match."
                ),
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class _EditFile:
    text: str
    had_bom: bool


def _validate_args(args: dict) -> tuple[str, str, str, bool] | str:
    path = args.get("path")
    if not isinstance(path, str) or not path.strip():
        return "[tool argument error: path must be a non-empty string]"

    old_text = args.get("old_text")
    if not isinstance(old_text, str) or not old_text:
        return "[tool argument error: old_text must be a non-empty string]"

    new_text = args.get("new_text")
    if not isinstance(new_text, str):
        return "[tool argument error: new_text must be a string]"

    if old_text == new_text:
        return "[tool argument error: old_text and new_text are identical; nothing to change]"

    replace_all = args.get("replace_all", False)
    if replace_all is None:
        replace_all = False
    if not isinstance(replace_all, bool):
        return "[tool argument error: replace_all must be a boolean]"

    return path.strip(), old_text, new_text, replace_all


def _resolve_edit_path(value: str) -> Path | str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return (
            f"[edit error: file not found: {value} — this tool edits existing "
            "files only; use run_command to create files]"
        )
    if not resolved.is_file():
        return f"[edit error: path is not a file: {value}]"
    return resolved


def _load_file(path: Path) -> _EditFile | str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"[edit error: could not read file: {exc}]"
    if len(data) > MAX_EDIT_FILE_BYTES:
        return (
            f"[edit error: file is {len(data)} bytes, exceeding the "
            f"{MAX_EDIT_FILE_BYTES} byte edit limit; use run_command for bulk edits]"
        )
    if b"\x00" in data:
        return f"[edit error: {path} appears to be a binary file]"
    had_bom = data.startswith(codecs.BOM_UTF8)
    if had_bom:
        data = data[len(codecs.BOM_UTF8):]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"[edit error: {path} is not valid UTF-8 text]"
    return _EditFile(text=text, had_bom=had_bom)


def _apply_replacement(
    text: str,
    old_text: str,
    new_text: str,
    replace_all: bool,
    *,
    path: Path,
) -> tuple[str, int] | str:
    count = text.count(old_text)
    if count == 0 and "\r\n" in text and "\n" in old_text and "\r" not in old_text:
        # The model usually emits LF; retry against CRLF files without
        # rewriting any other bytes of the file.
        crlf_old = old_text.replace("\n", "\r\n")
        crlf_count = text.count(crlf_old)
        if crlf_count:
            old_text = crlf_old
            new_text = new_text.replace("\n", "\r\n")
            count = crlf_count
    if count == 0:
        return (
            f"[edit error: old_text not found in {path}. Likely causes: "
            "whitespace or indentation differs from the file, the text spans "
            "lines with different line endings, or the file changed since you "
            "read it. Read the file and copy old_text exactly.]"
        )
    if count > 1 and not replace_all:
        return (
            f"[edit error: old_text matches {count} locations in {path}; "
            "include more surrounding lines to make it unique, or set "
            "replace_all=true]"
        )
    if replace_all:
        return text.replace(old_text, new_text), count
    return text.replace(old_text, new_text, 1), 1


def build_edit_diff(before: str, after: str, path: str) -> str:
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=path,
            tofile=path,
            lineterm="",
            n=_DIFF_CONTEXT_LINES,
        )
    )
    if len(diff_lines) > _MAX_DIFF_PREVIEW_LINES:
        hidden = len(diff_lines) - _MAX_DIFF_PREVIEW_LINES
        diff_lines = diff_lines[:_MAX_DIFF_PREVIEW_LINES]
        diff_lines.append(f"... {hidden} more diff lines ...")
    return "\n".join(diff_lines)


def _diff_renderable(diff_text: str) -> Group:
    lines = []
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---")):
            style = "dim"
        elif line.startswith("@@"):
            style = "cyan"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        else:
            style = "dim"
        lines.append(Text(line, style=style))
    return Group(*lines)


_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks"}
_SENSITIVE_NAMES = {
    "credentials",
    "id_rsa",
    "id_ed25519",
    "authorized_keys",
    "known_hosts",
    ".netrc",
    ".npmrc",
    ".pypirc",
}
_SENSITIVE_DIRS = {".ssh", ".gnupg", ".aws", ".azure", ".kube"}
_UNIX_SYSTEM_PREFIXES = ("/etc", "/usr", "/bin", "/sbin", "/boot", "/System", "/Library")


def _system_path_prefixes() -> list[str]:
    prefixes = [
        os.environ.get("SystemRoot", r"C:\Windows"),
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("ProgramData", r"C:\ProgramData"),
    ]
    prefixes.extend(_UNIX_SYSTEM_PREFIXES)
    return prefixes


def classify_edit(resolved: Path) -> tuple[bool, str]:
    """Return (risky, reason) for editing ``resolved``."""
    posix = str(resolved).replace("\\", "/").lower()
    for prefix in _system_path_prefixes():
        normalized = prefix.replace("\\", "/").lower().rstrip("/")
        if posix == normalized or posix.startswith(normalized + "/"):
            return True, "system path"

    name = resolved.name.lower()
    if name == ".env" or name.startswith(".env."):
        return True, "sensitive file (secrets/keys)"
    if resolved.suffix.lower() in _SENSITIVE_SUFFIXES:
        return True, "sensitive file (secrets/keys)"
    if name in _SENSITIVE_NAMES:
        return True, "sensitive file (secrets/keys)"

    parts = {part.lower() for part in resolved.parts[1:]}
    if parts & _SENSITIVE_DIRS:
        return True, "credentials directory"

    try:
        cwd = Path.cwd().resolve()
        in_cwd = resolved.is_relative_to(cwd)
    except OSError:
        in_cwd = False
    if not in_cwd:
        return True, "file outside the current working directory"

    # Only parts below the cwd count as hidden, so running jarv from inside a
    # dot-directory does not flag every file.
    if any(part.startswith(".") for part in resolved.relative_to(cwd).parts):
        return True, "hidden file or directory"

    return False, ""


def _check_edit(resolved: Path, diff_text: str, config: dict) -> tuple[bool, str]:
    """Gate an edit per command_safety. Returns (allowed, denial_message)."""
    level = get_setting(config, "command_safety")
    if level == "none":
        return True, ""

    if level == "all":
        reason = "all edits require approval"
    else:
        risky, reason = classify_edit(resolved)
        if not risky:
            return True, ""

    with approval_lock():
        body = Group(
            Text.from_markup(
                f"[bold yellow]⚠  File edit[/bold yellow]  [dim]—[/dim]  "
                f"[yellow]{escape(reason)}[/yellow]"
            ),
            Text(""),
            _diff_renderable(diff_text),
        )
        if prompt_panel_confirmation(
            body,
            subtitle="confirm to edit",
            question="Allow this edit?",
            kind="edit",
            reason=reason,
        ):
            return True, ""
    return False, f"[edit denied by user — {reason}]"


def _first_change_line(before: str, after: str) -> int:
    """1-based line number of the first differing line."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    for index, (old_line, new_line) in enumerate(zip(before_lines, after_lines)):
        if old_line != new_line:
            return index + 1
    return min(len(before_lines), len(after_lines)) + 1


def _format_result(path: Path, count: int, before: str, after: str) -> str:
    before_count = len(before.splitlines())
    after_count = len(after.splitlines())
    delta = after_count - before_count
    change_line = _first_change_line(before, after)

    after_lines = after.splitlines()
    start = max(0, change_line - 1 - _RESULT_CONTEXT_LINES)
    end = min(len(after_lines), change_line + _RESULT_CONTEXT_LINES)
    width = len(str(end)) if end else 1
    context = [
        f"  {number:>{width}} | {after_lines[number - 1]}"
        for number in range(start + 1, end + 1)
    ]

    lines = [
        "[EDIT RESULT]",
        f"Path: {path}",
        f"Replacements: {count}",
        f"Lines: {before_count} -> {after_count} ({delta:+d})",
        "Context (new file content around first change):",
    ]
    lines.extend(context)
    return "\n".join(lines)


def dispatch_edit_tool(
    args: dict,
    *,
    config: dict,
    cancellation_token: CancellationToken | None = None,
) -> ToolOutput:
    if not isinstance(args, dict):
        return "[tool argument error: edit arguments must be an object]"
    validated = _validate_args(args)
    if isinstance(validated, str):
        return validated
    value, old_text, new_text, replace_all = validated

    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()

    resolved = _resolve_edit_path(value)
    if isinstance(resolved, str):
        return resolved
    loaded = _load_file(resolved)
    if isinstance(loaded, str):
        return loaded

    replaced = _apply_replacement(
        loaded.text, old_text, new_text, replace_all, path=resolved
    )
    if isinstance(replaced, str):
        return replaced
    new_content, count = replaced

    diff_text = build_edit_diff(loaded.text, new_content, str(resolved))
    allowed, denial = _check_edit(resolved, diff_text, config)
    if not allowed:
        return denial

    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()

    data = new_content.encode("utf-8")
    if loaded.had_bom:
        data = codecs.BOM_UTF8 + data
    try:
        resolved.write_bytes(data)
    except OSError as exc:
        return f"[edit error: could not write file: {exc}]"

    return _format_result(resolved, count, loaded.text, new_content)
