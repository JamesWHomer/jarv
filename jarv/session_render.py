"""Pure rendering helpers for session history views."""

import json

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text

from .display import flatten_headings, output_renderable, rendered_text_lines, tool_card
from .tool_outputs import summarize_tool_output

def _history_content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
                elif "content" in c and isinstance(c["content"], str):
                    chunks.append(c["content"])
                else:
                    chunks.append(f"[{c.get('type', 'item')}]")
            else:
                chunks.append(str(c))
        return "\n".join(chunks)
    return str(content)


def _markdown_to_text_lines(content: str, width: int) -> list[Text]:
    return rendered_text_lines(Markdown(flatten_headings(content)), width)


def _status_renderable(item: dict) -> Text:
    content = _history_content_to_str(item.get("content", "")).strip()
    phase = str(item.get("phase", "")).lower()
    prefix = "\u2713 " if phase == "tool" else "\u2726 "
    return Text(f"{prefix}{content}", style="dim")


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
    for item in history[call_index + 1:]:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user" or item.get("type") == "function_call":
            break
        if (
            item.get("type") == "function_call_output"
            and item.get("call_id") == call_id
        ):
            return summarize_tool_output(item.get("output", ""))
    return ""


def _next_visible_history_item(history: list, start_index: int) -> dict | None:
    for candidate in history[start_index:]:
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


def _edit_snippet_lines(text: str, prefix: str, style: str) -> list[Text]:
    lines = text.splitlines() or [""]
    shown = [
        Text(prefix + line, style=style)
        for line in lines[:_EDIT_SNIPPET_MAX_LINES]
    ]
    hidden = len(lines) - _EDIT_SNIPPET_MAX_LINES
    if hidden > 0:
        shown.append(
            Text(f"  … {hidden} more line{'s' if hidden != 1 else ''}", style="dim italic")
        )
    return shown


def _edit_result_summary(output: str) -> str:
    """Condense an [EDIT RESULT] block into one line, e.g. '1 replacement  •  120 → 118 lines'."""
    replacements = ""
    lines_info = ""
    for line in output.splitlines():
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


def _tool_call_renderable(item: dict, output: str = "", *, display_mode: str = "fullscreen"):
    name = str(item.get("name") or "unknown")
    args, raw_arguments = _tool_call_arguments(item)
    failed = (
        args is None
        or output.startswith(
            (
                "[error:",
                "[tool argument error:",
                "[unknown tool:",
                "[edit error:",
                "[edit denied",
            )
        )
        or "cancelled by user" in output
    )
    status = "failed" if failed else "done"
    status_style = "red" if failed else "green"

    if name == "run_command" and args is not None:
        command_line = Text("> ", style="bold yellow")
        command_line.append(str(args.get("command", "")))
        body: object = command_line
        if output:
            body = Group(command_line, output_renderable(output))
        metadata = ""
        if isinstance(args.get("head_chars"), int) and isinstance(args.get("tail_chars"), int):
            metadata = (
                f"model window {args['head_chars']:,} / "
                f"{args['tail_chars']:,} chars"
            )
        return tool_card(
            name,
            body,
            metadata=metadata,
            status=status,
            status_style=status_style,
            display_mode=display_mode,
        )

    if name == "read" and args is not None:
        read_path = Text(
            str(args.get("input", "")),
            no_wrap=True,
            overflow="ellipsis",
        )
        read_meta = Text(
            f"offset {args.get('offset', 0)!r}  \u2022  "
            f"size {args.get('size', 'default')!r}",
            style="dim",
        )
        body = Group(read_path, read_meta)
    elif name == "edit" and args is not None:
        parts: list = [
            Text(str(args.get("path", "")), no_wrap=True, overflow="ellipsis")
        ]
        parts.extend(_edit_snippet_lines(str(args.get("old_text", "")), "- ", "red"))
        parts.extend(_edit_snippet_lines(str(args.get("new_text", "")), "+ ", "green"))
        if output.startswith("[EDIT RESULT]"):
            summary = _edit_result_summary(output)
            if summary:
                parts.append(Text(summary, style="dim"))
        elif output:
            parts.append(output_renderable(output))
        return tool_card(
            name,
            Group(*parts),
            metadata="replace all" if args.get("replace_all") else "",
            status=status,
            status_style=status_style,
            display_mode=display_mode,
        )
    elif name == "web_search" and args is not None:
        body = Text(str(args.get("query", "")))
        return tool_card(
            name,
            body,
            metadata="DuckDuckGo",
            status=status,
            status_style=status_style,
            display_mode=display_mode,
        )
    elif name == "ask_user" and args is not None:
        parts: list = [Markdown(flatten_headings(str(args.get("question", ""))))]
        if output:
            answer = Text("> ", style="bold cyan")
            answer.append(output)
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
                line.append("\u2713 ", style="bold green")
                line.append(label, style="bold cyan")
                if result.get("tldr"):
                    line.append(f"  {result['tldr']}", style="dim")
                lines.append(line)
        body = Group(*lines) if lines else Text(raw_arguments, style="dim")
    else:
        body = Text(raw_arguments, style="dim")

    return tool_card(
        name,
        body,
        status=status,
        status_style=status_style,
        display_mode=display_mode,
    )


def tool_call_card(item: dict, output: str = "", *, display_mode: str = "fullscreen"):
    """Render a tool call as a Rich card (shared by history and live UI)."""
    return _tool_call_renderable(item, output, display_mode=display_mode)


def tool_call_card_from_args(name: str, args: dict, *, display_mode: str = "fullscreen"):
    """Render a live tool card from parsed tool arguments."""
    return tool_call_card(
        {"name": name, "arguments": json.dumps(args, ensure_ascii=True)},
        display_mode=display_mode,
    )


def _history_visual_lines_and_anchors(history: list, width: int) -> tuple[list[Text], list[int]]:
    lines: list[Text] = []
    anchors: list[int] = []
    jarv_turn_open = False
    for item_index, item in enumerate(history):
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
        body = _history_content_to_str(item.get("content", "")).strip()
        if not body:
            continue
        start = len(lines)
        if role == "user":
            jarv_turn_open = False
            for j, raw in enumerate(body.splitlines() or [""]):
                t = Text(no_wrap=False, overflow="fold")
                if j == 0:
                    t.append("user: ", style="bold cyan")
                else:
                    t.append("  ")
                t.append(raw, style="bold")
                lines.extend(rendered_text_lines(t, width))
        elif role == "assistant":
            if not jarv_turn_open:
                lines.append(Text("jarv:", style="bold green", no_wrap=True, overflow="crop"))
            lines.extend(_markdown_to_text_lines(body, width))
            jarv_turn_open = False
        else:
            jarv_turn_open = False
            label = role or "?"
            for j, raw in enumerate(body.splitlines() or [""]):
                t = Text(no_wrap=False, overflow="fold")
                if j == 0:
                    t.append(f"{label}: ", style="dim")
                else:
                    t.append("  ")
                t.append(raw, style="dim")
                lines.extend(rendered_text_lines(t, width))
        if len(lines) > start:
            anchors.append(start)
        lines.append(Text(""))
    if lines and lines[-1].plain == "":
        lines.pop()
    return lines, anchors


def _history_visual_lines(history: list, width: int) -> list[Text]:
    lines, _ = _history_visual_lines_and_anchors(history, width)
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
