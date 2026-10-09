"""Pure rendering helpers for session history views."""

import json
import re

from rich.console import Group
from rich.text import Text

from .terminal_text import safe_terminal_text

from .display import (
    command_line_renderable,
    flatten_headings,
    hidden_lines_hint,
    output_renderable,
    rendered_text_lines,
    tool_card,
)
from .tool_outputs import (
    ToolOutcome,
    flatten_content_text,
    summarize_tool_output,
    tool_outcome,
    tool_output_failed,
    with_tool_outcome,
)

# Kept as an alias for heads-up and existing session-render callers.
_history_content_to_str = flatten_content_text


def _markdown_to_text_lines(content: str, width: int) -> list[Text]:
    from .markdown_render import markdown_renderable

    return rendered_text_lines(markdown_renderable(flatten_headings(content)), width)


def _status_renderable(item: dict) -> Text:
    content = _history_content_to_str(item.get("content", "")).strip()
    phase = str(item.get("phase", "")).lower()
    prefix = "\u2713 " if phase == "tool" else "\u2726 "
    return Text(safe_terminal_text(f"{prefix}{content}"), style="dim")


def _tool_call_arguments(item: dict) -> tuple[dict | None, str]:
    arguments = item.get("arguments", "")
    if not isinstance(arguments, str):
        return None, str(arguments)
    arguments = arguments.strip()
    if not arguments:
        return {}, ""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None, arguments
    if not isinstance(parsed, dict):
        return None, json.dumps(parsed, ensure_ascii=True)
    return parsed, json.dumps(parsed, ensure_ascii=True, separators=(", ", ": "))


def _tool_call_output(history: list, call_index: int, call_id) -> str:
    # Copying the whole suffix for each tool call makes long histories quadratic
    # even when the matching output is the very next item.
    for index in range(*slice(call_index + 1, None).indices(len(history))):
        item = history[index]
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user":
            break
        if (
            item.get("type") == "function_call_output"
            and item.get("call_id") == call_id
        ):
            output = summarize_tool_output(item.get("output", ""))
            if "outcome" in item:
                outcome = ToolOutcome.from_dict(item["outcome"]) or ToolOutcome("unknown")
                output = with_tool_outcome(output, outcome)
            return output
    return with_tool_outcome("", "unknown")


def _next_visible_history_item(history: list, start_index: int) -> dict | None:
    for index in range(*slice(start_index, None).indices(len(history))):
        candidate = history[index]
        if not isinstance(candidate, dict):
            continue
        if candidate.get("type") == "function_call":
            return candidate
        role = str(candidate.get("role", "")).lower()
        if role == "system":
            continue
        body = _history_content_to_str(candidate.get("content", "")).strip()
        if role and body:
            return candidate
    return None


_EDIT_SNIPPET_MAX_LINES = 3
_ARGS_PREVIEW_MAX_CHARS = 500


def _edit_snippet_lines(
    text: str, prefix: str, style: str, *, expanded: bool = False
) -> list[Text]:
    lines = safe_terminal_text(text).splitlines() or [""]
    cap = len(lines) if expanded else _EDIT_SNIPPET_MAX_LINES
    shown = [Text(prefix + line, style=style) for line in lines[:cap]]
    hidden = len(lines) - cap
    if hidden > 0:
        shown.append(Text("  ").append_text(hidden_lines_hint(hidden, where="below")))
    return shown


def _format_byte_size(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.0f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


_READ_HEADER_END_RE = re.compile(r"\r?\n\r?\n")


def _read_result_summary(output: str) -> str:
    """Condense a [READ RESULT] header into one line, e.g. '4,096 of 45,120 chars · more available'.

    Parses header lines only — the content chunk after the first blank line is
    never touched, so file contents can never leak into the card.
    """
    if not output.startswith("[READ RESULT]"):
        return ""
    # Tool cards summarize this header on every paint. Avoid sanitizing and
    # splitting the read body (up to 200,000 characters) just to discard it.
    boundary = _READ_HEADER_END_RE.search(output)
    header = output[:boundary.start()] if boundary is not None else output
    returned = total = image_bytes = None
    eof = media_type = ""
    for line in safe_terminal_text(header).splitlines():
        if not line.strip():
            break
        for label, target in (
            ("Returned size: ", "returned"),
            ("Total size: ", "total"),
            ("Image bytes: ", "image_bytes"),
        ):
            if line.startswith(label):
                value = line.removeprefix(label).strip()
                if value.isdigit():
                    if target == "returned":
                        returned = int(value)
                    elif target == "total":
                        total = int(value)
                    else:
                        image_bytes = int(value)
        if line.startswith("EOF: "):
            eof = line.removeprefix("EOF: ").strip()
        elif line.startswith("Image media type: "):
            media_type = line.removeprefix("Image media type: ").strip()
    if media_type:
        size = f"  •  {_format_byte_size(image_bytes)}" if image_bytes is not None else ""
        return f"image {media_type}{size}"
    if returned is None:
        return ""
    if eof == "true" or total is None:
        return f"{returned:,} chars  •  EOF"
    return f"{returned:,} of {total:,} chars  •  more available"


def _read_result_content(output: str) -> str:
    """The content chunk of a [READ RESULT] block (after the header's blank
    line) — shown only when a card is explicitly expanded. Image reads have no
    displayable content, so they return empty."""
    if not output.startswith("[READ RESULT]"):
        return ""
    header, separator, content = output.partition("\n\n")
    if not separator or "Image media type: " in header:
        return ""
    return content.strip("\n")


_WEB_RESULT_TITLE_RE = re.compile(r"^\d+\. (.+)$")


def _web_search_result_summary(output: str) -> tuple[str, list[str]]:
    """Parse '<N> results' and the top result titles from web_search output."""
    titles: list[str] = []
    count = 0
    for line in safe_terminal_text(output).splitlines():
        match = _WEB_RESULT_TITLE_RE.match(line)
        if match is None:
            continue
        count += 1
        if len(titles) < 3:
            titles.append(match.group(1))
    if count == 0:
        return "", []
    return f"{count} result{'s' if count != 1 else ''}", titles


def _read_args_metadata(args: dict) -> str:
    parts: list[str] = []
    offset = args.get("offset")
    if isinstance(offset, int) and not isinstance(offset, bool) and offset:
        parts.append(f"offset {offset:,}")
    size = args.get("size")
    if isinstance(size, int) and not isinstance(size, bool):
        parts.append(f"size {size:,}")
    return "  •  ".join(parts)


def _edit_result_summary(output: str) -> str:
    """Condense an [EDIT RESULT] block into one line, e.g. '1 replacement  •  120 → 118 lines'."""
    replacements = ""
    lines_info = ""
    for line in safe_terminal_text(output).splitlines():
        if line.startswith("Replacements: "):
            replacements = line.removeprefix("Replacements: ").strip()
        elif line.startswith("Lines: "):
            lines_info = line.removeprefix("Lines: ").strip()
    parts = []
    if replacements:
        suffix = "" if replacements == "1" else "s"
        parts.append(f"{replacements} replacement{suffix}")
    if lines_info:
        before, _, rest = lines_info.partition(" -> ")
        after = rest.split(" ")[0] if rest else ""
        if before and after:
            parts.append(f"{before} → {after} lines")
    return "  •  ".join(parts)


def _error_line(output: str) -> Text:
    first = safe_terminal_text(output).splitlines()[0] if output else ""
    return Text(first, style="dim red")


def _tool_call_renderable(
    item: dict,
    output: str = "",
    *,
    display_mode: str = "fullscreen",
    expanded: bool = False,
    status_override: tuple[str, str] | None = None,
):
    """Render a tool call as one card: header (icon \u00b7 metadata \u00b7 status pill),
    an input summary, and an optional result preview. Every tool goes through
    the single ``tool_card`` call at the end so the card contract cannot drift
    per tool."""
    name = str(item.get("name") or "unknown")
    args, raw_arguments = _tool_call_arguments(item)
    outcome = tool_outcome(output)
    failed = tool_output_failed(output)
    if outcome is None or outcome.status == "unknown":
        failed = failed or args is None
    status = "failed" if failed else "done"
    status_style = "red" if failed else "green"
    if not failed and outcome is not None and outcome.status in {"running", "unknown"}:
        status = outcome.status
        status_style = "yellow"
    if status_override is not None:
        status, status_style = status_override
    metadata = ""

    if name == "run_command" and args is not None:
        command_line = command_line_renderable(
            str(args.get("command", "")), expanded=expanded
        )
        body: object = command_line
        if output:
            body = Group(
                command_line,
                output_renderable(
                    output, display_mode=display_mode, expanded=expanded
                ),
            )
    elif name == "read" and args is not None:
        parts: list = [
            Text(safe_terminal_text(str(args.get("input", ""))), no_wrap=True, overflow="ellipsis")
        ]
        if failed and output:
            parts.append(_error_line(output))
        else:
            summary = _read_result_summary(output)
            if summary:
                parts.append(Text(safe_terminal_text(summary), style="dim"))
            if expanded:
                content = _read_result_content(output)
                if content:
                    parts.append(Text(safe_terminal_text(content), style="dim"))
        body = Group(*parts)
        metadata = _read_args_metadata(args)
    elif name == "edit" and args is not None:
        parts = [Text(safe_terminal_text(str(args.get("path", ""))), no_wrap=True, overflow="ellipsis")]
        parts.extend(
            _edit_snippet_lines(
                str(args.get("old_text", "")), "- ", "red", expanded=expanded
            )
        )
        parts.extend(
            _edit_snippet_lines(
                str(args.get("new_text", "")), "+ ", "green", expanded=expanded
            )
        )
        if output.startswith("[EDIT RESULT]"):
            summary = _edit_result_summary(output)
            if summary:
                parts.append(Text(safe_terminal_text(summary), style="dim"))
        elif failed and output:
            parts.append(_error_line(output))
        elif output:
            parts.append(
                output_renderable(output, display_mode=display_mode, expanded=expanded)
            )
        body = Group(*parts)
        metadata = "replace all" if args.get("replace_all") else ""
    elif name == "web_search" and args is not None:
        parts = [Text(safe_terminal_text(str(args.get("query", ""))))]
        if failed and output:
            parts.append(_error_line(output))
        elif expanded and output:
            parts.append(Text(safe_terminal_text(output), style="dim"))
        else:
            summary, titles = _web_search_result_summary(output)
            if summary:
                parts.append(Text(safe_terminal_text(summary), style="dim"))
                for title in titles:
                    parts.append(
                        Text(safe_terminal_text(f"  {title}"), style="dim", no_wrap=True, overflow="ellipsis")
                    )
        body = Group(*parts)
        from .web import SEARCH_ENGINE_LABEL

        metadata = SEARCH_ENGINE_LABEL
    elif name == "ask_user" and args is not None:
        from rich.markdown import Markdown

        parts = [Markdown(flatten_headings(safe_terminal_text(str(args.get("question", "")))))]
        if output:
            answer = Text("> ", style="bold cyan")
            answer.append(safe_terminal_text(output))
            parts.append(answer)
        body = Group(*parts)
    elif name == "spawn" and args is not None:
        result_by_label: dict[str, dict] = {}
        try:
            results = json.loads(output) if output else []
        except json.JSONDecodeError:
            results = []
        if isinstance(results, list):
            result_by_label = {
                str(result.get("label")): result
                for result in results
                if isinstance(result, dict) and result.get("label")
            }
        lines: list[Text] = []
        children = args.get("children", [])
        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict):
                    continue
                label = str(child.get("label", "?"))
                result = result_by_label.get(label, {})
                line = Text()
                child_status = result.get("status")
                if child_status == "done":
                    line.append("\u2713 ", style="bold green")
                elif child_status == "failed":
                    line.append("\u2717 ", style="bold red")
                else:
                    line.append("? ", style="yellow")
                line.append(safe_terminal_text(label), style="bold cyan")
                if result.get("tldr"):
                    line.append(safe_terminal_text(f"  {result['tldr']}"), style="dim")
                elif result.get("reason"):
                    line.append(safe_terminal_text(f"  {result['reason']}"), style="dim red")
                lines.append(line)
        body = Group(*lines) if lines else Text(safe_terminal_text(raw_arguments), style="dim")
    else:
        if not expanded and len(raw_arguments) > _ARGS_PREVIEW_MAX_CHARS:
            args_text = Text(safe_terminal_text(raw_arguments[:_ARGS_PREVIEW_MAX_CHARS]), style="dim")
            args_text.append(
                f" … +{len(raw_arguments) - _ARGS_PREVIEW_MAX_CHARS:,} chars",
                style="dim italic",
            )
        else:
            args_text = Text(safe_terminal_text(raw_arguments), style="dim")
        body = args_text
        if output:
            body = Group(
                body,
                output_renderable(output, display_mode=display_mode, expanded=expanded),
            )

    return tool_card(
        name,
        body,
        metadata=metadata,
        status=status,
        status_style=status_style,
        display_mode=display_mode,
    )


class ToolCallCard:
    """Lazy tool card that retains its raw call data so a transcript entry can
    be re-rendered expanded (full command + output) on demand. Renders through
    ``_tool_call_renderable`` on every paint, so it drops into every place a
    pre-built card renderable was used."""

    expandable = True

    def __init__(self, item: dict, output: str, display_mode: str):
        self.item = item
        self.output = output
        self.display_mode = display_mode
        self.expanded = False

    def __rich_console__(self, console, options):
        yield _tool_call_renderable(
            self.item,
            self.output,
            display_mode=self.display_mode,
            expanded=self.expanded,
        )


def tool_call_card(item: dict, output: str = "", *, display_mode: str = "fullscreen"):
    """Render a tool call as a Rich card (shared by history and live UI)."""
    return ToolCallCard(item, output, display_mode)


def tool_call_card_from_args(
    name: str,
    args: dict,
    *,
    output: str = "",
    display_mode: str = "fullscreen",
):
    """Render a live tool card from parsed tool arguments."""
    return tool_call_card(
        {"name": name, "arguments": json.dumps(args, ensure_ascii=True)},
        output,
        display_mode=display_mode,
    )


def _history_visual_lines_and_anchors(history: list, width: int, *, cancelled=None) -> tuple[list[Text], list[int]]:
    lines: list[Text] = []
    anchors: list[int] = []
    jarv_turn_open = False
    for item_index, item in enumerate(history):
        if cancelled is not None and cancelled():
            return [], []
        if not isinstance(item, dict):
            continue
        if item.get("type") == "status":
            body = _history_content_to_str(item.get("content", "")).strip()
            if not body:
                continue
            if not jarv_turn_open:
                lines.append(Text("jarv:", style="bold green", no_wrap=True, overflow="crop"))
                jarv_turn_open = True
            lines.extend(rendered_text_lines(_status_renderable(item), width))
            continue
        if item.get("type") == "function_call":
            start = len(lines)
            if not jarv_turn_open:
                lines.append(Text("jarv:", style="bold green", no_wrap=True, overflow="crop"))
                jarv_turn_open = True
            output = _tool_call_output(
                history,
                item_index,
                item.get("call_id"),
            )
            lines.extend(
                rendered_text_lines(
                    _tool_call_renderable(item, output),
                    width,
                )
            )
            if len(lines) > start:
                anchors.append(start)
            next_item = _next_visible_history_item(history, item_index + 1)
            if not isinstance(next_item, dict) or next_item.get("type") != "function_call":
                lines.append(Text(""))
            continue
        role = str(item.get("role", "")).lower()
        if role == "system":
            continue
        body = safe_terminal_text(_history_content_to_str(item.get("content", ""))).strip()
        if not body:
            continue
        start = len(lines)
        if role == "assistant":
            if not jarv_turn_open:
                lines.append(Text("jarv:", style="bold green", no_wrap=True, overflow="crop"))
            lines.extend(_markdown_to_text_lines(body, width))
            jarv_turn_open = False
        else:
            jarv_turn_open = False
            label = role or "?"
            label_style = "bold cyan" if role == "user" else "dim"
            body_style = "bold" if role == "user" else "dim"
            for j, raw in enumerate(body.splitlines() or [""]):
                t = Text(no_wrap=False, overflow="fold")
                if j == 0:
                    t.append(f"{label}: ", style=label_style)
                else:
                    t.append("  ")
                t.append(raw, style=body_style)
                lines.extend(rendered_text_lines(t, width))
        if len(lines) > start:
            anchors.append(start)
        lines.append(Text(""))
    if lines and lines[-1].plain == "":
        lines.pop()
    return lines, anchors


def _history_visual_lines(history: list, width: int, *, cancelled=None) -> list[Text]:
    lines, _ = _history_visual_lines_and_anchors(history, width, cancelled=cancelled)
    return lines


def _session_row_widths(width: int) -> tuple[int, int, int]:
    """Allocate session, date, and message columns within a row."""
    date_width = min(7, max(0, width))
    if width <= date_width:
        return (0, date_width, 0)

    gutter_width = 2
    message_min_width = 16
    fixed_width = date_width + (2 * gutter_width)
    session_width = min(28, max(0, width - fixed_width - message_min_width))
    message_width = max(0, width - session_width - fixed_width)
    return (session_width, date_width, message_width)
