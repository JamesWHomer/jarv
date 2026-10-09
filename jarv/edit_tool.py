"""Exact string-replacement editing of local text files."""

from __future__ import annotations

import codecs
import difflib
import os
import stat
import tempfile
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from rich.console import Group
from rich.markup import escape
from rich.text import Text

from .cancellation import CancellationToken
from .config import get_setting
from .safety import approval_lock, prompt_panel_confirmation
from .terminal_text import safe_terminal_text
from .tool_outputs import ToolOutput, with_tool_outcome


MAX_EDIT_FILE_BYTES = 5_000_000
_DIFF_CONTEXT_LINES = 3
_MAX_DIFF_PREVIEW_LINES = 60
_MAX_DIFF_PREVIEW_CHARS = 12_000
_MAX_RESULT_PREVIEW_CHARS = 4_000
_RESULT_CONTEXT_LINES = 3
_FILE_LOCKS = weakref.WeakValueDictionary()
_FILE_LOCKS_GUARD = threading.Lock()


@contextmanager
def _file_edit_lock(path: Path, token: CancellationToken | None):
    # Resolved paths make relative paths and symlink aliases share a lock.
    key = os.path.normcase(str(path))
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[key] = lock
    while not lock.acquire(timeout=0.1):
        if token is not None:
            token.throw_if_cancelled()
    try:
        if token is not None:
            token.throw_if_cancelled()
        yield
    finally:
        lock.release()


def _revision(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_mode, info.st_ctime_ns)


def _conflict(path: Path) -> str:
    return (f"[edit conflict: {path} changed since it was read; "
            "no edit was written. Read the file and retry.]")

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
    data: bytes
    revision: tuple
    mode: int


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


def _resolve_edit_path(value: str, *, cwd: str | Path | None = None) -> Path | str:
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path(cwd) if cwd is not None else Path.cwd()) / path
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return (
            f"[edit error: file not found: {value} — this tool edits existing "
            "files only; use run_command to create files]"
        )
    if not resolved.is_file():
        return f"[edit error: path is not a file: {value}]"
    return resolved


def _load_file(path: Path) -> _EditFile | str:
    try:
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            data = source.read(MAX_EDIT_FILE_BYTES + 1)
            after = os.fstat(source.fileno())
        if _revision(before) != _revision(after):
            return _conflict(path)
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
    original_data = data
    if had_bom:
        data = data[len(codecs.BOM_UTF8):]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"[edit error: {path} is not valid UTF-8 text]"
    return _EditFile(text=text, had_bom=had_bom, data=original_data,
                     revision=_revision(after), mode=stat.S_IMODE(after.st_mode))


def _commit_edit(path: Path, loaded: _EditFile, data: bytes,
                 token: CancellationToken | None) -> str | None:
    temporary = None
    try:
        # Same-directory staging keeps replacement on the same filesystem.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".jarv-edit-",
                                         delete=False) as staged:
            temporary = Path(staged.name)
            staged.write(data)
            staged.flush()
            os.fsync(staged.fileno())
        temporary.chmod(loaded.mode)
        if token is not None:
            token.throw_if_cancelled()
        try:
            with path.open("rb") as current:
                before = os.fstat(current.fileno())
                current_data = current.read(MAX_EDIT_FILE_BYTES + 1)
                after = os.fstat(current.fileno())
            if (current_data != loaded.data
                    or _revision(before) != loaded.revision
                    or _revision(after) != loaded.revision
                    # Windows stat/fstat can report different ctime meanings.
                    or _revision(path.lstat())[:-1] != loaded.revision[:-1]):
                return _conflict(path)
        except OSError:
            return _conflict(path)
        # The per-path lock excludes sibling Jarv edits through replacement.
        # External writers do not honor it; portable replace is not a CAS.
        os.replace(temporary, path)
    except OSError as exc:
        return f"[edit error: could not write file: {exc}]"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return None


def _apply_replacement(
    text: str,
    old_text: str,
    new_text: str,
    replace_all: bool,
    *,
    path: Path,
    source_bytes: int | None = None,
) -> tuple[str, int] | str:
    count = text.count(old_text)
    if count == 0 and "\r\n" in text and "\n" in old_text and "\r" not in old_text:
        # The model usually emits LF; retry against CRLF files without
        # rewriting any other bytes of the file.
        crlf_old = old_text.replace("\n", "\r\n")
        crlf_count = text.count(crlf_old)
        if crlf_count:
            old_text = crlf_old
            new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
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
    # Check the encoded result before str.replace can multiply a short
    # replacement across millions of matches. source_bytes includes any BOM;
    # the strings here already reflect the CRLF fallback above.
    try:
        original_size = _utf8_size(text) if source_bytes is None else source_bytes
        result_size = original_size + (count if replace_all else 1) * (
            _utf8_size(new_text) - _utf8_size(old_text)
        )
    except UnicodeEncodeError:
        return "[tool argument error: replacement text must be valid UTF-8 text]"
    if result_size > MAX_EDIT_FILE_BYTES:
        return (
            f"[edit error: replacement would produce {result_size} bytes, exceeding "
            f"the {MAX_EDIT_FILE_BYTES} byte edit limit; use run_command for bulk edits]"
        )
    if replace_all:
        return text.replace(old_text, new_text), count
    return text.replace(old_text, new_text, 1), 1


def _utf8_size(text: str) -> int:
    return sum(len(text[start:start + 64 * 1024].encode("utf-8"))
               for start in range(0, len(text), 64 * 1024))


def _bounded_preview(lines, *, max_chars: int, max_lines: int | None = None) -> str:
    """Bound previews by characters as well as lines, including minified files."""
    marker = "\n... preview truncated ..."
    parts = []
    used = 0
    for index, line in enumerate(lines):
        if max_lines is not None and index >= max_lines:
            return "\n".join(parts) + marker
        remaining = max_chars - len(marker) - used - bool(parts)
        if len(line) > remaining:
            parts.append(line[:max(0, remaining)])
            return "\n".join(parts) + marker
        used += len(line) + bool(parts)
        parts.append(line)
    return "\n".join(parts)


def build_edit_diff(before: str, after: str, path: str) -> str:
    def source_lines(text: str) -> list[str]:
        # Only real LF/CRLF delimit lines. splitlines() also consumes standalone
        # CR and C1 controls, potentially making a changed file look identical.
        lines = text.replace("\r\n", "\n").split("\n")
        if lines[-1] == "":
            lines.pop()
        return lines

    diff_lines = difflib.unified_diff(
        source_lines(before),
        source_lines(after),
        fromfile=path,
        tofile=path,
        lineterm="",
        n=_DIFF_CONTEXT_LINES,
    )
    # Compare original contents, then escape each display row before joining:
    # an original trailing CR must not merge with our added LF into CRLF.
    return _bounded_preview((safe_terminal_text(line) for line in diff_lines),
                            max_chars=_MAX_DIFF_PREVIEW_CHARS,
                            max_lines=_MAX_DIFF_PREVIEW_LINES)


def _diff_renderable(diff_text: str) -> Group:
    lines = []
    for line in safe_terminal_text(diff_text).splitlines():
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


def classify_edit(resolved: Path, *, cwd: str | Path | None = None) -> tuple[bool, str]:
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
        cwd = (Path(cwd) if cwd is not None else Path.cwd()).resolve()
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


def _check_edit(
    resolved: Path, diff_text: str, config: dict, *, cwd: str | Path | None = None,
    cancellation_token: CancellationToken | None = None,
) -> tuple[bool, str]:
    """Gate an edit per command_safety. Returns (allowed, denial_message)."""
    control = config.get("_run_control")
    if cancellation_token is None and control is not None:
        cancellation_token = control.token
    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()
    level = get_setting(config, "command_safety")
    if level == "none":
        return True, ""

    if level == "all":
        reason = "all edits require approval"
    else:
        risky, reason = classify_edit(resolved, cwd=cwd)
        if not risky:
            return True, ""

    from .run_control import require_user_input
    require_user_input(config, f"Manual approval required for edit: {reason}.")
    cancel_kwargs = {"cancellation_token": cancellation_token} if cancellation_token is not None else {}
    lock = approval_lock()
    while not lock.acquire(timeout=0.05):
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
    try:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        body = Group(
            Text.from_markup(
                f"[bold yellow]⚠  File edit[/bold yellow]  [dim]—[/dim]  "
                f"[yellow]{escape(reason)}[/yellow]"
            ),
            Text(""),
            _diff_renderable(diff_text),
        )
        approved = prompt_panel_confirmation(
            body,
            subtitle="confirm to edit",
            question="Allow this edit?",
            kind="edit",
            reason=reason,
            **cancel_kwargs,
        )
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        if approved:
            return True, ""
    finally:
        lock.release()
    return False, f"[edit denied by user — {reason}]"


def _format_result(path: Path, count: int, before: str, after: str) -> str:
    # Reuse the split lines for counts, the first change, and its context.
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    before_count = len(before_lines)
    after_count = len(after_lines)
    delta = after_count - before_count
    change_line = min(before_count, after_count) + 1
    for number, (old_line, new_line) in enumerate(zip(before_lines, after_lines), 1):
        if old_line != new_line:
            change_line = number
            break

    start = max(0, change_line - 1 - _RESULT_CONTEXT_LINES)
    end = min(len(after_lines), change_line + _RESULT_CONTEXT_LINES)
    width = len(str(end)) if end else 1
    context = (
        f"  {number:>{width}} | {after_lines[number - 1][:_MAX_RESULT_PREVIEW_CHARS + 1]}"
        for number in range(start + 1, end + 1)
    )

    lines = [
        "[EDIT RESULT]",
        f"Path: {path}",
        f"Replacements: {count}",
        f"Lines: {before_count} -> {after_count} ({delta:+d})",
        "Context (new file content around first change):",
    ]
    return _bounded_preview(chain(lines, context), max_chars=_MAX_RESULT_PREVIEW_CHARS)


def dispatch_edit_tool(
    args: dict,
    *,
    config: dict,
    cancellation_token: CancellationToken | None = None,
    cwd: str | Path | None = None,
) -> ToolOutput:
    if not isinstance(args, dict):
        return with_tool_outcome("[tool argument error: edit arguments must be an object]", "failed")
    validated = _validate_args(args)
    if isinstance(validated, str):
        return with_tool_outcome(validated, "failed")
    value, old_text, new_text, replace_all = validated

    control = config.get("_run_control")
    if cancellation_token is None and control is not None:
        cancellation_token = control.token
    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()

    resolved = _resolve_edit_path(value, cwd=cwd)
    if isinstance(resolved, str):
        return with_tool_outcome(resolved, "failed")
    with _file_edit_lock(resolved, cancellation_token):
        return _edit_locked(resolved, old_text, new_text, replace_all,
                            config, cancellation_token, cwd=cwd)


def _edit_locked(resolved: Path, old_text: str, new_text: str, replace_all: bool,
                 config: dict, cancellation_token: CancellationToken | None,
                 *, cwd: str | Path | None = None) -> ToolOutput:
    loaded = _load_file(resolved)
    if isinstance(loaded, str):
        return with_tool_outcome(loaded, "failed")

    replaced = _apply_replacement(
        loaded.text, old_text, new_text, replace_all, path=resolved,
        source_bytes=len(loaded.data),
    )
    if isinstance(replaced, str):
        return with_tool_outcome(replaced, "failed")
    new_content, count = replaced

    diff_text = build_edit_diff(loaded.text, new_content, str(resolved))
    allowed, denial = _check_edit(
        resolved, diff_text, config, cwd=cwd, cancellation_token=cancellation_token,
    )
    if not allowed:
        return with_tool_outcome(denial, "denied")

    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()

    data = new_content.encode("utf-8")
    if loaded.had_bom:
        data = codecs.BOM_UTF8 + data
    error = _commit_edit(resolved, loaded, data, cancellation_token)
    if error is not None:
        return with_tool_outcome(error, "failed")

    return with_tool_outcome(_format_result(resolved, count, loaded.text, new_content), "success")
