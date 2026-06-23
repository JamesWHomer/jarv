"""Interactive shell-command UI: terminal-control parsing, cards, and continuation.

Extracted from agent.py so the root run loop stays focused on orchestration. These
helpers turn a model's single-line terminal reply into a process action, render the
per-interaction command card, and format the prompts/outputs exchanged with the model
during an interactive `run_command`.
"""

from __future__ import annotations

from rich.console import Group
from rich.text import Text

from .agent_ui import _ui_call
from .config import DEFAULT_CONFIG
from .display import console, tool_card
from .orchestrator import PendingRunCommand
from .shell import command_result_renderable


def _format_elapsed_seconds(seconds: float | int | None) -> str:
    try:
        value = max(0.0, float(seconds or 0.0))
    except (TypeError, ValueError):
        value = 0.0
    if value < 10:
        return f"{value:.1f}s"
    return f"{value:.0f}s"


def _interactive_check_in_seconds(config: dict) -> float | None:
    try:
        seconds = float(config.get("command_timeout", 60))
    except (TypeError, ValueError):
        return 60.0
    return seconds if seconds > 0 else None


def _run_command_waiting_prompt(snapshot) -> str:
    result = snapshot.to_delta_command_result()
    status_lines = []
    if getattr(snapshot, "check_in", False):
        status_lines.extend([
            "Status: command_timeout check-in; the process is still running and was not killed.",
            f"Elapsed: {_format_elapsed_seconds(getattr(snapshot, 'elapsed_seconds', 0.0))}",
            f"Time since last output: {_format_elapsed_seconds(getattr(snapshot, 'idle_seconds', 0.0))}",
        ])
    header = (
        "[interactive command still running]\n"
        if getattr(snapshot, "check_in", False)
        else "[interactive command waiting for terminal input]\n"
    )
    status_text = "\n".join(status_lines) + "\n\n" if status_lines else ""
    return (
        f"{header}"
        f"Command: {snapshot.command}\n\n"
        f"{status_text}"
        "New command output since last interaction:\n"
        "```text\n"
        f"{result.full_model_output()}\n"
        "```\n\n"
        "Reply with exactly one line of terminal input or one control. Do not "
        "explain. Do not chain multiple controls. Plain text sends that text "
        "followed by Enter. Special controls: <ENTER>, <WAIT>, <WAIT 10s>, "
        "<CTRL_C>, <EOF>, <CTRL_D>, <ESC>, <TAB>, <UP>, <DOWN>, <LEFT>, "
        "<RIGHT>. Only the first input/control will be used."
    )


def _run_command_final_prompt(output: str) -> str:
    return (
        "[interactive command exited]\n"
        "Final command output:\n"
        "```text\n"
        f"{output}\n"
        "```"
    )


_TERMINAL_CONTROL_NAMES = (
    "<ENTER>",
    "<WAIT",
    "<CTRL_C>",
    "<EOF>",
    "<CTRL_D>",
    "<ESC>",
    "<TAB>",
    "<UP>",
    "<DOWN>",
    "<LEFT>",
    "<RIGHT>",
)


def _first_terminal_action(text: str) -> str:
    lines = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if line.lower().startswith("stdin>"):
            line = line[6:].strip()
        lines.append(line)
    if not lines:
        return ""

    line = lines[0]
    upper = line.upper()
    first_control_at: int | None = None
    for token in _TERMINAL_CONTROL_NAMES:
        idx = upper.find(token)
        if idx >= 0 and (first_control_at is None or idx < first_control_at):
            first_control_at = idx

    if first_control_at is None:
        return line
    if first_control_at > 0:
        return line[:first_control_at].strip()

    closing = line.find(">")
    if closing >= 0:
        return line[:closing + 1].strip()
    return line


def _parse_terminal_control(text: str) -> tuple[str, str | float | None]:
    stripped = _first_terminal_action(text)
    upper = stripped.upper()
    if upper == "<ENTER>":
        return "stdin", "\n"
    if upper == "<WAIT>":
        return "wait", None
    if upper.startswith("<WAIT ") and upper.endswith(">"):
        raw = stripped[6:-1].strip().lower()
        multiplier = 1.0
        if raw.endswith("ms"):
            multiplier = 0.001
            raw = raw[:-2].strip()
        elif raw.endswith("s"):
            raw = raw[:-1].strip()
        try:
            return "wait", max(0.0, float(raw) * multiplier)
        except ValueError:
            return "stdin", stripped.rstrip("\n") + "\n"
    if upper == "<CTRL_C>":
        return "ctrl_c", None
    if upper in {"<EOF>", "<CTRL_D>"}:
        return "eof", None
    special_keys = {
        "<ESC>": "\x1b",
        "<TAB>": "\t",
        "<UP>": "\x1b[A",
        "<DOWN>": "\x1b[B",
        "<RIGHT>": "\x1b[C",
        "<LEFT>": "\x1b[D",
    }
    if upper in special_keys:
        return "stdin_raw", special_keys[upper]
    return "stdin", stripped.rstrip("\n") + "\n"


def _terminal_action_display(action: str, payload: str | float | None) -> str:
    if action == "stdin":
        text = str(payload or "").rstrip("\n")
        return text or "<ENTER>"
    if action == "wait":
        if isinstance(payload, float):
            return f"<WAIT {payload:g}s>"
        return "<WAIT>"
    if action == "ctrl_c":
        return "<CTRL_C>"
    if action == "eof":
        return "<EOF>"
    if action == "stdin_raw":
        controls = {
            "\x1b": "<ESC>",
            "\t": "<TAB>",
            "\x1b[A": "<UP>",
            "\x1b[B": "<DOWN>",
            "\x1b[C": "<RIGHT>",
            "\x1b[D": "<LEFT>",
        }
        return controls.get(str(payload or ""), "<KEY>")
    return ""


def _interactive_command_card(
    prepared,
    snapshot,
    config: dict,
    *,
    status: str,
    terminal_reply: str | None = None,
):
    display_mode = config.get(
        "tool_call_display",
        DEFAULT_CONFIG["tool_call_display"],
    )
    command_line = Text("> ", style="bold yellow")
    command_line.append(prepared.cmd)
    body_parts = [command_line]
    if terminal_reply is not None:
        stdin_line = Text("stdin> ", style="bold cyan")
        stdin_line.append(terminal_reply.rstrip("\n") or "<ENTER>")
        body_parts.append(stdin_line)
    result = snapshot.to_delta_command_result()
    if status == "waiting":
        if getattr(snapshot, "check_in", False):
            body_parts.append(Text(
                "command_timeout check-in; process still running",
                style="dim",
            ))
            body_parts.append(Text(
                "elapsed "
                f"{_format_elapsed_seconds(getattr(snapshot, 'elapsed_seconds', 0.0))}"
                " | idle "
                f"{_format_elapsed_seconds(getattr(snapshot, 'idle_seconds', 0.0))}",
                style="dim",
            ))
        body_parts.append(Text(result.full_model_output()))
        body_parts.append(Text("Waiting for terminal input", style="dim"))
    else:
        body_parts.append(command_result_renderable(result))
    return tool_card(
        "run_command",
        Group(*body_parts),
        metadata=(
            f"model window {prepared.head_chars:,} / "
            f"{prepared.tail_chars:,} chars"
        ),
        display_mode=display_mode,
        status=status,
        status_style="blue" if status == "waiting" else (
            "green" if result.exit_code in (None, 0) else "red"
        ),
    )


def _show_interactive_command_card(
    pending: PendingRunCommand,
    snapshot,
    config: dict,
    *,
    status: str,
    terminal_reply: str | None = None,
    ui=None,
) -> None:
    card = _interactive_command_card(
        pending.prepared,
        snapshot,
        config,
        status=status,
        terminal_reply=terminal_reply,
    )
    if ui is not None:
        _ui_call(ui, "show_tool_card", card)
        return
    console.print(card)
    if config.get(
        "tool_call_display",
        DEFAULT_CONFIG["tool_call_display"],
    ) == "print":
        console.print()


def _format_finished_interactive_output(
    pending: PendingRunCommand,
    snapshot,
    retained_store,
) -> str:
    from .orchestrator import format_run_command_output

    result = snapshot.to_delta_command_result()
    output, _output_id = format_run_command_output(
        result,
        pending.prepared,
        retained_store,
    )
    if callable(pending.unregister_cancel):
        pending.unregister_cancel()
        pending.unregister_cancel = None
    return output


def _continue_interactive_command(
    pending: PendingRunCommand,
    terminal_reply: str,
    *,
    config: dict,
    cancellation_token=None,
):
    action, payload = _parse_terminal_control(terminal_reply)
    action_display = _terminal_action_display(action, payload)
    check_in_seconds = _interactive_check_in_seconds(config)
    if action == "stdin":
        pending.process.write_stdin(str(payload))
        snapshot = pending.process.wait_until_idle(
            check_in_seconds=check_in_seconds,
            cancellation_token=cancellation_token,
        )
    elif action == "stdin_raw":
        pending.process.write_stdin(str(payload or ""))
        snapshot = pending.process.wait_until_idle(
            check_in_seconds=check_in_seconds,
            cancellation_token=cancellation_token,
        )
    elif action == "wait":
        snapshot = pending.process.wait_until_idle(
            wait_seconds=payload if isinstance(payload, float) else 0.5,
            check_in_seconds=check_in_seconds,
            cancellation_token=cancellation_token,
        )
    elif action == "ctrl_c":
        pending.process.interrupt()
        snapshot = pending.process.wait_until_idle(
            check_in_seconds=check_in_seconds,
            cancellation_token=cancellation_token,
        )
    elif action == "eof":
        pending.process.close_stdin()
        snapshot = pending.process.wait_until_idle(
            check_in_seconds=check_in_seconds,
            cancellation_token=cancellation_token,
        )
    else:
        snapshot = pending.process.wait_until_idle(
            check_in_seconds=check_in_seconds,
            cancellation_token=cancellation_token,
        )
    return snapshot, action_display
