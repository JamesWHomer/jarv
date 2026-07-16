"""Interactive shell-command UI: terminal-control parsing, cards, and continuation.

Extracted from agent.py so the root run loop stays focused on orchestration. These
helpers turn a model's single-line terminal reply into a process action, render the
per-interaction command card, and format the prompts/outputs exchanged with the model
during an interactive `run_command`.
"""

from __future__ import annotations

import difflib
import re
import time
from contextlib import contextmanager

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


def _interactive_check_in_seconds(config: dict) -> float:
    try:
        seconds = float(config.get("command_timeout", 60))
    except (TypeError, ValueError):
        return 60.0
    # command_timeout <= 0 means "never kill", but a check-in must still fire
    # eventually or a never-idle process would hold wait_until_idle forever.
    return seconds if seconds > 0 else 300.0


def _interactive_max_rounds(config: dict) -> int:
    try:
        return max(1, int(config.get("interactive_max_rounds", 40)))
    except (TypeError, ValueError):
        return 40


# Single source of truth for the terminal control vocabulary shown to the model,
# reused by the waiting prompt's help line and the tool-call reminder below.
_TERMINAL_CONTROLS = (
    "<ENTER>, <WAIT>, <WAIT Ns|Nm> (e.g. <WAIT 30s>, <WAIT 2m>), <CTRL_C>, "
    "<EOF>, <CTRL_D>, <CTRL_x> (any letter), <ESC>, <TAB>, <SPACE>, "
    "<BACKSPACE>, <DELETE>, <HOME>, <END>, <PAGE_UP>, <PAGE_DOWN>, "
    "<UP>, <DOWN>, <LEFT>, <RIGHT>"
)

# Sent once per session; later waiting prompts omit it and carry only state.
_TERMINAL_REPLY_INSTRUCTIONS = (
    "Reply with one line — stdin text (sent with Enter) or controls: "
    f"{_TERMINAL_CONTROLS}. Controls may be chained in order (e.g. "
    "<DOWN> <DOWN> <ENTER>); an empty reply just waits. Do not mix prose with "
    "controls — text around controls is not sent. In a multi-line reply, a "
    "line containing only controls is executed and surrounding prose is "
    "ignored."
)


def _interactive_tool_call_reminder() -> str:
    """Nudge sent when the model calls a tool mid-interaction instead of replying."""
    return (
        "A terminal command is waiting for input — reply with stdin text or "
        f"controls ({_TERMINAL_CONTROLS}), not a tool call."
    )


def _run_command_waiting_prompt(
    snapshot,
    *,
    include_help: bool = True,
    prepared=None,
    statuses: tuple = (),
) -> str:
    result = snapshot.to_delta_command_result()
    # The interactive path must respect the same head/tail window as one-shot
    # commands, or a chatty process injects its whole backlog into the prompt.
    if prepared is not None:
        output = result.to_model_output(
            head_chars=getattr(prepared, "head_chars", None),
            tail_chars=getattr(prepared, "tail_chars", None),
        )
    else:
        output = result.full_model_output()
    status_lines = list(statuses)
    if getattr(snapshot, "check_in", False):
        status_lines.extend([
            "Status: command_timeout check-in; the process is still running and was not killed.",
            f"Elapsed: {_format_elapsed_seconds(getattr(snapshot, 'elapsed_seconds', 0.0))}",
            f"Time since last output: {_format_elapsed_seconds(getattr(snapshot, 'idle_seconds', 0.0))}",
        ])
    if getattr(snapshot, "stdin_closed", False):
        status_lines.append(
            "Status: stdin is closed — typed text can no longer be delivered; "
            "use <WAIT>, <CTRL_C>, or let the process exit."
        )
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
        f"{output}\n"
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


# Cap on controls executed from one reply; keeps a runaway chain bounded.
_MAX_ACTIONS_PER_REPLY = 8

_TOKEN_RE = re.compile(r"<[A-Za-z_][^<>]*>")

# Raw VT byte sequences written to the process's stdin. That stdin is an
# anonymous pipe (see shell.py), not a ConPTY: line-oriented readers will not
# see these as keys, but raw-mode readers (menus, pagers, REPLs) will — the
# same delivery the arrow keys have always used.
_SPECIAL_KEYS = {
    "<ESC>": "\x1b",
    "<TAB>": "\t",
    "<SPACE>": " ",
    "<BACKSPACE>": "\x08",
    "<DELETE>": "\x1b[3~",
    "<HOME>": "\x1b[H",
    "<END>": "\x1b[F",
    "<PAGE_UP>": "\x1b[5~",
    "<PAGE_DOWN>": "\x1b[6~",
    "<UP>": "\x1b[A",
    "<DOWN>": "\x1b[B",
    "<RIGHT>": "\x1b[C",
    "<LEFT>": "\x1b[D",
}

_SPECIAL_KEY_NAMES = {sequence: name for name, sequence in _SPECIAL_KEYS.items()}

# Accepted <WAIT ...> unit suffixes, longest first so "ms" wins over "s" and
# "min" over "m"; a bare number means seconds.
_WAIT_UNITS = (("ms", 0.001), ("sec", 1.0), ("min", 60.0), ("s", 1.0), ("m", 60.0))

_CTRL_LETTER_RE = re.compile(r"<CTRL_([A-Z])>")


def _normalize_token_name(token: str) -> str:
    """Canonicalize a ``<...>`` token: uppercase, ``-``/spaces to ``_``.

    Models spell controls as ``<Ctrl-C>`` or ``<PAGE DOWN>`` about as often as
    the advertised forms; rejecting those spellings only burns a re-prompt.
    """
    inner = token[1:-1] if token.startswith("<") and token.endswith(">") else token
    return "<" + re.sub(r"[\s-]+", "_", inner.strip()).upper() + ">"


def _control_token_action(token: str) -> tuple[str, str | float | None] | None:
    """Parse one ``<...>`` token into an action, or None when it isn't a control."""
    upper = token.upper()
    # WAIT is matched before separator normalization because its payload
    # legitimately contains a space.
    if upper == "<WAIT>":
        return "wait", None
    if upper.startswith("<WAIT") and upper.endswith(">"):
        raw = token[5:-1].strip().lower()
        multiplier = 1.0
        for suffix, factor in _WAIT_UNITS:
            if raw.endswith(suffix):
                multiplier = factor
                raw = raw[: -len(suffix)].strip()
                break
        try:
            return "wait", max(0.0, float(raw) * multiplier)
        except ValueError:
            # Malformed wait: surfaced as a re-prompt, never typed into stdin.
            return "invalid", token
    name = _normalize_token_name(token)
    if name == "<ENTER>":
        return "stdin", "\n"
    if name == "<CTRL_C>":
        return "ctrl_c", None
    if name in {"<EOF>", "<CTRL_D>"}:
        return "eof", None
    if name in _SPECIAL_KEYS:
        return "stdin_raw", _SPECIAL_KEYS[name]
    ctrl = _CTRL_LETTER_RE.fullmatch(name)
    if ctrl:
        # Down a pipe these are plain bytes with no signal semantics — which
        # is why C (interrupt) and D (EOF) keep their special cases above.
        return "stdin_raw", chr(ord(ctrl.group(1)) - 64)
    return None


def _candidate_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if line.lower().startswith("stdin>"):
            line = line[6:].strip()
        lines.append(line)
    return lines


def _control_line_redirect(
    later_lines: list[str],
) -> tuple[list[tuple[str, str | float | None]], str | None] | None:
    """Find the first later line that is nothing but valid control tokens.

    Reasoning models often narrate on line one and put the actual control on
    the next line; typing that narration into a live process is the worst
    possible reading of the reply. Only pure-control lines redirect — a later
    line mixing prose and controls is no clearer than line one, and a
    malformed control (``<WAIT soon>``) invalidates the reply outright: the
    model clearly meant a control, so re-prompt rather than type prose.
    """
    for line in later_lines:
        matches = list(_TOKEN_RE.finditer(line))
        if not matches or _TOKEN_RE.sub("", line).strip():
            continue
        parsed = [_control_token_action(match.group()) for match in matches]
        for action in parsed:
            if action is not None and action[0] == "invalid":
                return [action], None
        if any(action is None for action in parsed):
            continue
        return (
            parsed[:_MAX_ACTIONS_PER_REPLY],
            "your reply had multiple lines; the control line was executed "
            "and the surrounding prose was not typed into stdin",
        )
    return None


def _parse_terminal_actions(
    text: str,
) -> tuple[list[tuple[str, str | float | None]], str | None]:
    """Parse a model reply into an ordered action sequence plus an advisory note.

    Rules (first non-fence line, with one multi-line exception below):
    - an empty reply is a wait, never Enter (Enter can confirm destructive prompts);
    - no control tokens -> the line is stdin text (Enter implied), except a line
      that is nothing but unknown ``<...>`` tokens, which is invalid (re-prompt);
    - multi-line exception: a multi-word first line with no valid control is
      treated as narration when a later line is pure controls — that control
      line is executed instead of typing the narration into stdin. A one-word
      first line ("y", "3") is a real answer and never redirected;
    - a single word followed by controls (``3<WAIT>``) -> the word is the stdin
      answer and the trailing controls are dropped;
    - otherwise controls win: the run of controls starting at the first one is
      executed in order and surrounding prose is dropped, with a note so the
      model learns what was ignored;
    - a malformed control (``<WAIT soon>``) invalidates the reply (re-prompt).
    """
    lines = _candidate_lines(text)
    if not lines:
        return [("wait", None)], None
    line = lines[0]

    matches = [
        (match, _control_token_action(match.group()))
        for match in _TOKEN_RE.finditer(line)
    ]
    controls = [(match, action) for match, action in matches if action is not None]
    if not controls:
        if matches and not _TOKEN_RE.sub("", line).strip():
            return [("invalid", line)], None
        if any(ch.isspace() for ch in line):
            redirect = _control_line_redirect(lines[1:])
            if redirect is not None:
                return redirect
        return [("stdin", line.rstrip("\n") + "\n")], None

    first_match = controls[0][0]
    prefix = line[:first_match.start()].strip()
    if prefix and not any(ch.isspace() for ch in prefix):
        return (
            [("stdin", prefix + "\n")],
            "the controls after your stdin text were ignored — send controls on their own",
        )

    note = "the text before your controls was not sent to stdin" if prefix else None
    actions: list[tuple[str, str | float | None]] = []
    position = first_match.start()
    for match, action in controls:
        if line[position:match.start()].strip():
            note = (
                "only the leading run of controls was executed; "
                "the rest of the line was ignored"
            )
            break
        if action[0] == "invalid":
            return [action], None
        actions.append(action)
        position = match.end()
        if len(actions) >= _MAX_ACTIONS_PER_REPLY:
            break
    if line[position:].strip() and note is None:
        note = "the text after your controls was not sent to stdin"
    return actions, note


# Known non-WAIT controls, for close-match suggestions in rejection messages.
_CONTROL_VOCABULARY = (
    "<ENTER>", "<CTRL_C>", "<EOF>", "<CTRL_D>", *_SPECIAL_KEYS,
)


def _invalid_reply_message(payload: str) -> str:
    """Explain a rejected reply, naming the exact offending token.

    The old generic ``could not parse '...'`` left models re-sending the same
    bad token round after round; naming the token and the closest valid
    control (or the literal-text fallback for single letters) converges the
    retry loop instead.
    """
    hints = []
    for token in _TOKEN_RE.findall(payload):
        action = _control_token_action(token)
        if action is not None and action[0] != "invalid":
            continue
        if token.upper().startswith("<WAIT"):
            hints.append(
                f"{token} is not a valid wait — use <WAIT Ns> or <WAIT Nm>, "
                "e.g. <WAIT 30s> or <WAIT 2m>"
            )
            continue
        inner = token[1:-1].strip()
        if len(inner) == 1 and inner.isalpha():
            hints.append(
                f"{token} is not a control — to type '{inner.lower()}', "
                f"reply with the literal text {inner.lower()}"
            )
            continue
        close = difflib.get_close_matches(
            _normalize_token_name(token), _CONTROL_VOCABULARY, n=1
        )
        if close:
            hints.append(
                f"{token} is not a supported control — did you mean {close[0]}?"
            )
        else:
            hints.append(f"{token} is not a supported control")
    if not hints:
        return f"could not parse {payload!r}"
    return f"could not parse {payload!r}: " + "; ".join(hints)


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
        sequence = str(payload or "")
        name = _SPECIAL_KEY_NAMES.get(sequence)
        if name:
            return name
        if len(sequence) == 1 and 1 <= ord(sequence) <= 26:
            return f"<CTRL_{chr(ord(sequence) + 64)}>"
        return "<KEY>"
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
    Ctrl-C, EOF, chained sequences) reuse :func:`_interaction_marker`'s wording
    in a dim style so they read as side notes rather than typed input. Used by
    the growing :class:`~jarv.agent_ui.InteractiveCommandCard` for each
    per-step line.
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


def _record_interactive_input(
    pending,
    action: str,
    action_display: str,
    output_delta: str = "",
) -> None:
    """Append one sent input/control (and the output it produced) to the record.

    Earlier versions replaced the stored output with the marker list, so a
    checkpoint taken mid-interaction lost every byte the command had printed;
    the record now grows marker/output interleaved and survives cancellation.
    """
    marker = _interaction_marker(action, action_display)
    pending.input_markers.append(marker)
    pending.transcript_segments.append(marker)
    if output_delta and output_delta.strip() and output_delta.strip() != "(no output)":
        pending.transcript_segments.append(output_delta)
    output_item = getattr(pending, "output_item", None)
    if isinstance(output_item, dict):
        output_item["output"] = "\n".join(pending.transcript_segments)


# Consecutive unparseable replies before the interaction is aborted; bounded
# separately from interactive_max_rounds (see PendingRunCommand.invalid_streak).
_MAX_CONSECUTIVE_INVALID = 5


def _record_rejected_reply(pending, marker: str) -> None:
    """Append a rejected/blocked-reply marker line to the collapsed record.

    Rejected replies used to live only in the un-persisted live conversation,
    so a stalled interaction left no trace in the session sidecar; the marker
    makes the failure diagnosable after the fact.
    """
    pending.transcript_segments.append(marker)
    output_item = getattr(pending, "output_item", None)
    if isinstance(output_item, dict):
        output_item["output"] = "\n".join(pending.transcript_segments)


def _finalize_interactive_record(pending, final_output: str) -> None:
    """Close the record: the interleaved transcript followed by the final output."""
    output_item = getattr(pending, "output_item", None)
    if not isinstance(output_item, dict):
        return
    segments = list(pending.transcript_segments)
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


def _cancellable_sleep(seconds: float, cancellation_token=None) -> None:
    deadline = time.monotonic() + max(0.0, min(seconds, 60.0))
    while time.monotonic() < deadline:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        time.sleep(0.05)


@contextmanager
def _suspended_card_live(pending):
    """Pause the held-open inline card ``Live`` around a console prompt.

    The card's auto-refresh thread would repaint over the y/N prompt and Rich
    hides the cursor while a Live is active, so the user would type blind.
    No-op when a confirm handler owns the display (heads-up: the prompt never
    touches the console) or in heads-up's inline-card-less mode
    (``pending.live is None``). stop()/start(refresh=True) reuse is the same
    pattern as ``AltScreenApp.suspended``.
    """
    from .safety import confirm_handler_active

    live = getattr(pending, "live", None)
    if live is None or confirm_handler_active():
        yield
        return
    try:
        live.stop()
    except Exception:
        pass
    try:
        yield
    finally:
        try:
            live.start(refresh=True)
        except Exception:
            pass


def _screen_stdin_text(pending, actions, config: dict) -> str | None:
    """Local safety gate for typed stdin lines.

    ``check_command`` screens only the initial command string; text typed into
    a shell/REPL afterwards used to bypass it entirely. Classify each typed
    line with the same risky patterns and confirm locally (no LLM auditor —
    stdin lines are cheap to deny and may be prompt answers, not commands).
    Under ``command_safety="all"`` every non-empty typed line needs approval,
    or an approved bare shell/REPL would be an ungated escape hatch; controls,
    raw keys, and plain Enter never prompt.
    """
    level = config.get("command_safety", "risky")
    if level == "none":
        return None
    from .safety import approval_lock, classify_command, prompt_confirmation

    for action, payload in actions:
        if action != "stdin":
            continue
        text = str(payload or "").strip()
        if not text:
            continue
        is_risky, reason = classify_command(text)
        if not is_risky:
            if level != "all":
                continue
            reason = "all commands require approval"
        with approval_lock(), _suspended_card_live(pending):
            approved = prompt_confirmation(
                f"stdin to `{pending.prepared.cmd}`: {text}",
                reason,
                kind="stdin",
                question="Allow this input?",
            )
        if not approved:
            if not is_risky:
                return "[stdin blocked by user — safety level is set to 'all']"
            return f"[stdin blocked by user — detected as risky: {reason}]"
    return None


def _continue_interactive_command(
    pending: PendingRunCommand,
    terminal_reply: str,
    *,
    config: dict,
    cancellation_token=None,
):
    """Parse and apply one model reply to the held-open process.

    Returns ``(snapshot, action_display, action_kind, note)``. ``snapshot`` is
    ``None`` when nothing touched the process (malformed reply or a blocked
    stdin line); ``action_display`` then carries the message for the model.
    """
    actions, note = _parse_terminal_actions(terminal_reply)
    if actions[0][0] == "invalid":
        return None, _invalid_reply_message(str(actions[0][1])), "invalid", note
    denial = _screen_stdin_text(pending, actions, config)
    if denial is not None:
        return None, denial, "blocked", note

    process = pending.process
    check_in_seconds = _interactive_check_in_seconds(config)

    def apply(action: str, payload) -> None:
        if action in ("stdin", "stdin_raw"):
            process.write_stdin(str(payload or ""))
        elif action == "ctrl_c":
            process.interrupt()
        elif action == "eof":
            process.close_stdin()
        elif action == "wait":
            _cancellable_sleep(
                payload if isinstance(payload, float) else 2.0,
                cancellation_token,
            )

    for action, payload in actions[:-1]:
        apply(action, payload)

    last_action, last_payload = actions[-1]
    if last_action == "wait":
        if isinstance(last_payload, float):
            snapshot = process.wait_until_idle(
                wait_seconds=last_payload,
                check_in_seconds=check_in_seconds,
                cancellation_token=cancellation_token,
            )
        else:
            # Bare <WAIT>: hold until output arrives and settles or the
            # check-in fires, instead of a near-useless fixed 0.5s poll that
            # cost a whole model round-trip per tick.
            snapshot = process.wait_until_idle(
                check_in_seconds=check_in_seconds,
                cancellation_token=cancellation_token,
                require_output=True,
            )
    else:
        apply(last_action, last_payload)
        snapshot = process.wait_until_idle(
            check_in_seconds=check_in_seconds,
            cancellation_token=cancellation_token,
        )

    if len(actions) == 1:
        kind = last_action
        display = _terminal_action_display(last_action, last_payload)
    else:
        kind = "sequence"
        display = " ".join(_terminal_action_display(a, p) for a, p in actions)
    return snapshot, display, kind, note
