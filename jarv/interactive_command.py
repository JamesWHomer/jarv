"""Interactive shell-command UI: terminal-control parsing, cards, and continuation.

Extracted from agent.py so the root run loop stays focused on orchestration. These
helpers turn a model's single-line terminal reply into a process action, render the
per-interaction command card, and format the prompts/outputs exchanged with the model
during an interactive `run_command`.
"""

from __future__ import annotations

from rich.text import Text

from .orchestrator import PendingRunCommand


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


# Single source of truth for the terminal control vocabulary shown to the model,
# reused by the waiting prompt's help line and the tool-call reminder below.
_TERMINAL_CONTROLS = (
    "<ENTER>, <WAIT>, <WAIT Ns> (e.g. <WAIT 30s>), <CTRL_C>, <EOF>, "
    "<CTRL_D>, <ESC>, <TAB>, <UP>, <DOWN>, <LEFT>, <RIGHT>"
)

# Sent once per session; later waiting prompts omit it and carry only state.
_TERMINAL_REPLY_INSTRUCTIONS = (
    "Reply with one line — stdin text (sent with Enter) or one control: "
    f"{_TERMINAL_CONTROLS}. Only the first line is used."
)


def _interactive_tool_call_reminder() -> str:
    """Nudge sent when the model calls a tool mid-interaction instead of replying."""
    return (
        "A terminal command is waiting for input — reply with stdin text or one "
        f"control ({_TERMINAL_CONTROLS}), not a tool call."
    )


def _run_command_waiting_prompt(snapshot, *, include_help: bool = True) -> str:
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
    help_text = f"\n\n{_TERMINAL_REPLY_INSTRUCTIONS}" if include_help else ""
    return (
        f"{header}"
        f"Command: {snapshot.command}\n\n"
        f"{status_text}"
        "New command output since last interaction:\n"
        "```text\n"
        f"{result.full_model_output()}\n"
        "```"
        f"{help_text}"
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


def _terminal_action_is_stdin(action: str) -> bool:
    """Whether the action actually wrote characters to the process's stdin."""
    return action in {"stdin", "stdin_raw"}


def _interaction_marker(action: str, action_display: str) -> str:
    """One-line record of what was sent, distinguishing stdin from controls.

    Waits, Ctrl-C and EOF never reach stdin, so labelling them ``stdin>`` (as
    the card used to) is misleading; they get their own dim markers instead.
    """
    if _terminal_action_is_stdin(action):
        return f"stdin> {action_display or '<ENTER>'}"
    if action == "wait":
        inner = action_display.strip("<>").strip().lower() or "wait"
        return f"[{inner}]"
    if action == "ctrl_c":
        return "[sent Ctrl-C]"
    if action == "eof":
        return "[closed stdin / EOF]"
    return f"[{action_display}]"


def _interaction_marker_text(action: str, action_display: str) -> Text:
    """Styled one-line marker for what was sent this step.

    stdin writes get a bright ``stdin>`` prefix; non-stdin controls (waits,
    Ctrl-C, EOF) reuse :func:`_interaction_marker`'s wording in a dim style so
    they read as side notes rather than typed input. Used by the growing
    :class:`~jarv.agent_ui.InteractiveCommandCard` for each per-step line.
    """
    if _terminal_action_is_stdin(action):
        line = Text("stdin> ", style="bold cyan")
        line.append((action_display or "").rstrip("\n") or "<ENTER>")
        return line
    return Text(_interaction_marker(action, action_display), style="dim cyan")


def _attach_interactive_output_item(pending, history: list) -> None:
    """Point a pending command at its stored ``function_call_output`` item.

    The whole interactive exchange is collapsed into that single tool record so
    the resumed/reloaded transcript shows one command card instead of a stream
    of repeated waiting prompts and ``[terminal input sent]`` chat messages.
    """
    call_id = getattr(pending, "call_id", None)
    for item in reversed(history):
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id") == call_id
        ):
            pending.output_item = item
            break


def _record_interactive_input(pending, action: str, action_display: str) -> None:
    """Append one sent input/control to the single stored command record."""
    output_item = getattr(pending, "output_item", None)
    if not isinstance(output_item, dict):
        return
    pending.input_markers.append(_interaction_marker(action, action_display))
    output_item["output"] = "\n".join(pending.input_markers)


def _finalize_interactive_record(pending, final_output: str) -> None:
    """Collapse a finished interaction into inputs followed by the final output."""
    output_item = getattr(pending, "output_item", None)
    if not isinstance(output_item, dict):
        return
    segments: list[str] = []
    if pending.input_markers:
        segments.append("\n".join(pending.input_markers))
    if final_output and final_output.strip():
        segments.append(final_output)
    output_item["output"] = "\n".join(segments)


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
    return snapshot, action_display, action
