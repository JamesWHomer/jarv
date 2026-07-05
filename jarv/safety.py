import re
import sys
import threading
import time
from pathlib import Path

from rich.console import Group
from rich.markup import escape
from rich.text import Text

from .cancellation import CancellationToken, TurnCancelled
from .display import console, jarv_panel, live_display_depth

SAFETY_LEVELS = ("all", "risky", "none")
DEFAULT_SAFETY_LEVEL = "risky"

# Serialize interactive approval/audit prompts across concurrent subagent workers.
_APPROVAL_LOCK = threading.Lock()

# Each pattern is (compiled_regex, description) for user-facing confirmation.
# Patterns are designed to work across Windows (PowerShell/cmd) and Unix shells.
_RISKY_PATTERNS: list[tuple[re.Pattern, str]] = []


def _p(pattern: str, description: str) -> None:
    _RISKY_PATTERNS.append((re.compile(pattern, re.IGNORECASE), description))


# ── Destructive filesystem operations ─────────────────────────────────────
_p(r"\brm\s+(-[^\s]*[rf]|--recursive|--force)", "recursive/forced file deletion (rm)")
_p(r"\brmdir\s+/s\b", "recursive directory deletion (rmdir /s)")
_p(r"\bdel\s+.*/[sqf]", "forced file deletion (del)")
_p(r"\bRemove-Item\b.*-(Recurse|Force)", "recursive/forced file deletion (Remove-Item)")
_p(r"\bRemove-Item\b.*\s-[rfRF]\b", "recursive/forced file deletion (Remove-Item -r/-f)")
_p(r"\b(ri|rm)\s+.*\s-[rfRF]\b", "recursive/forced file deletion (PowerShell alias)")
_p(r"\b(python|python3|py)\s+(-c|--command)\b", "inline Python execution")
_p(r"\bshred\b", "secure file destruction (shred)")
_p(r"\bwipe\b", "disk/file wiping")

# ── Disk / partition operations ───────────────────────────────────────────
_p(r"\bformat\s+[a-zA-Z]:", "drive formatting (format)")
_p(r"\bFormat-Volume\b", "volume formatting (Format-Volume)")
_p(r"\bmkfs\b", "filesystem creation (mkfs)")
_p(r"\bfdisk\b", "partition editing (fdisk)")
_p(r"\bparted\b", "partition editing (parted)")
_p(r"\bdiskpart\b", "disk partitioning (diskpart)")
_p(r"\bdd\s+if=", "raw disk copy (dd)")

# ── Privilege escalation ──────────────────────────────────────────────────
_p(r"\bsudo\b", "elevated privileges (sudo)")
_p(r"\bdoas\b", "elevated privileges (doas)")
_p(r"\brunas\b", "elevated privileges (runas)")
_p(r"\bchmod\s+[0-7]*[67][0-7]{2}\b", "broad permission grant (chmod)")
_p(r"\bchmod\s+.*[ugoa]*\+[rwxsStX]*[sS]", "setuid/setgid (chmod +s)")
_p(r"\bchown\b", "ownership change (chown)")
_p(r"\bicacls\b.*(/grant|/remove|/deny)", "permission change (icacls)")

# ── Network exfiltration / remote code execution ─────────────────────────
_p(r"\bcurl\b.*\|\s*(sudo\s+)?(bash|sh|zsh|dash|powershell|pwsh)\b", "remote code execution (curl | shell)")
_p(r"\bwget\b.*\|\s*(sudo\s+)?(bash|sh|zsh|dash|powershell|pwsh)\b", "remote code execution (wget | shell)")
_p(r"\bwget\b.*-O\s*-.*\|", "remote code execution (wget -O - | ...)")
_p(r"\b(Invoke-WebRequest|Invoke-RestMethod|irm|iwr)\b.*\|\s*(Invoke-Expression|iex)\b", "remote code execution (IWR | IEX)")
_p(r"\b(iex|Invoke-Expression)\s*\(.*\b(Invoke-WebRequest|irm|iwr|Invoke-RestMethod)\b", "remote code execution (iex + IWR)")
_p(r"\bInvoke-Expression\b", "dynamic code execution (Invoke-Expression)")
_p(r"\biex\s+[^|]", "dynamic code execution (iex)")
_p(r"\bnc\b\s+.*-[el]", "netcat listener/exec")
_p(r"\bncat\b", "ncat network connection")
_p(r"\bscp\b", "secure copy to remote (scp)")
_p(r"\brsync\b.*[^/]\w+@\w+:", "remote sync (rsync to remote host)")

# ── System modification ──────────────────────────────────────────────────
_p(r"\breg\s+(delete|add)\b", "registry modification (reg)")
_p(r"\bNew-ItemProperty\b.*Registry", "registry modification (PowerShell)")
_p(r"\bRemove-ItemProperty\b.*Registry", "registry modification (PowerShell)")
_p(r"\bsystemctl\s+(disable|stop|mask)\b", "service control (systemctl)")
_p(r"\blaunchctl\s+(unload|remove)\b", "service control (launchctl)")
_p(r"\bschtasks\s+/(create|delete)\b", "scheduled task modification (schtasks)")
_p(r"\bcrontab\s+-[re]", "cron job modification (crontab)")

# ── Process / service killing ────────────────────────────────────────────
_p(r"\btaskkill\b", "process termination (taskkill)")
_p(r"\bkill\s+-9\b", "forced process kill (kill -9)")
_p(r"\bkillall\b", "mass process termination (killall)")
_p(r"\bpkill\b", "pattern-based process kill (pkill)")
_p(r"\bStop-Process\b", "process termination (Stop-Process)")
_p(r"\bStop-Service\b", "service stop (Stop-Service)")

# ── Package manager (global / system-wide) ───────────────────────────────
_p(r"\b(pip|pip3)\s+install\b(?!.*--user)(?!.*-e\s+\.)(?!.*--target)", "global pip install")
_p(r"\b(python|python3)\s+-m\s+pip\s+install\b(?!.*--user)(?!.*-e\s+\.)(?!.*--target)", "global pip install")
_p(r"\bnpm\s+(install|i)\s+.*-g\b|\bnpm\s+.*-g\s+(install|i)\b", "global npm install")
_p(r"\bchoco\s+(install|uninstall)\b", "Chocolatey package management")
_p(r"\bwinget\s+(install|uninstall)\b", "winget package management")
_p(r"\bbrew\s+(install|uninstall|remove)\b", "Homebrew package management")
_p(r"\bapt(-get)?\s+(install|remove|purge)\b", "apt package management")
_p(r"\byum\s+(install|remove|erase)\b", "yum package management")
_p(r"\bdnf\s+(install|remove|erase)\b", "dnf package management")
_p(r"\bpacman\s+-[SRU]", "pacman package management")

# ── Credential / secret access ───────────────────────────────────────────
_p(r"\b(cat|less|more|head|tail)\s+.*\.(env|pem|key|p12|pfx|jks)\b", "reading secrets file")
_p(r"\btype\s+.*\.(env|pem|key|p12|pfx)\b", "reading secrets file")
_p(r"\bGet-Content\b.*\.(env|pem|key|p12|pfx)\b", "reading secrets file")
_p(r"\b(cat|type|less|more|Get-Content)\b.*[/\\]\.ssh[/\\]", "reading SSH keys")
_p(r"\bssh-keygen\b", "SSH key generation")
_p(r"\b(cat|type|Get-Content)\b.*[/\\]\.gnupg[/\\]", "reading GPG keys")

# ── Git destructive operations ───────────────────────────────────────────
_p(r"\bgit\s+push\s+.*(-f\b|--force\b)(?!.*--force-with-lease)", "force push (git push --force)")
_p(r"\bgit\s+reset\s+--hard\b", "hard reset (git reset --hard)")
_p(r"\bgit\s+clean\s+-[^\s]*f", "forced clean (git clean -f)")
_p(r"\bgit\s+checkout\s+--\s+\.", "discard all changes (git checkout -- .)")
_p(r"\bgit\s+branch\s+-D\b", "force delete branch (git branch -D)")

# ── Environment / shell manipulation ─────────────────────────────────────
_p(r"\bexport\s+(PATH|LD_PRELOAD|LD_LIBRARY_PATH)=", "environment variable modification")
_p(r"\bsetx\s+(PATH|PATHEXT)\b", "permanent environment modification (setx)")
_p(r"\$env:(PATH|PATHEXT)\s*=", "environment modification (PowerShell)")
_p(r"\bSet-ExecutionPolicy\b", "PowerShell execution policy change")
_p(r"\beval\s+\$\(", "dynamic shell evaluation (eval)")
_p(r"\bsource\s+/dev/stdin\b", "sourcing from stdin")


def classify_command(command: str) -> tuple[bool, str]:
    """Check whether a command matches any risky pattern.

    Returns (is_risky, description).  description is empty when not risky.
    """
    for pattern, description in _RISKY_PATTERNS:
        if pattern.search(command):
            return True, description
    return False, ""


_CONTEXT_LINES_CAP = 4


def _build_confirmation_body(command: str, reason: str) -> Group:
    """Build the rich panel body, highlighting only risky lines."""
    reason_line = Text.from_markup(
        f"[bold yellow]\u26a0  Risky command[/bold yellow]  [dim]\u2014[/dim]  [yellow]{escape(reason)}[/yellow]"
    )

    lines = [ln for ln in command.splitlines() if ln.strip()]

    # Single-line command \u2014 show as before
    if len(lines) <= 1:
        command_line = Text.assemble(
            ("$ ", "bold bright_black"),
            (command.strip(), "bold bright_white"),
        )
        return Group(reason_line, Text(""), command_line)

    # Multi-line: classify each line, show risky ones prominently
    risky_indices: list[int] = []
    for i, ln in enumerate(lines):
        is_risky, _ = classify_command(ln)
        if is_risky:
            risky_indices.append(i)

    parts: list[Text | str] = [reason_line, Text("")]

    # Context before risky lines
    if risky_indices:
        first_risky = risky_indices[0]
        context_before = lines[:first_risky]
        if context_before:
            if len(context_before) <= _CONTEXT_LINES_CAP:
                for ln in context_before:
                    parts.append(Text.assemble(("  ", ""), (ln, "dim")))
            else:
                for ln in context_before[:2]:
                    parts.append(Text.assemble(("  ", ""), (ln, "dim")))
                hidden = len(context_before) - 3
                parts.append(Text.from_markup(
                    f"  [dim italic]\u2026 {hidden} more line{'s' if hidden != 1 else ''} \u2026[/dim italic]"
                ))
                parts.append(Text.assemble(("  ", ""), (context_before[-1], "dim")))
            parts.append(Text(""))

        # Risky lines
        for idx in risky_indices:
            parts.append(Text.assemble(
                ("  $ ", "bold bright_black"),
                (lines[idx], "bold bright_white"),
            ))

        # Context after last risky line
        last_risky = risky_indices[-1]
        context_after = lines[last_risky + 1:]
        if context_after:
            parts.append(Text(""))
            if len(context_after) <= _CONTEXT_LINES_CAP:
                for ln in context_after:
                    parts.append(Text.assemble(("  ", ""), (ln, "dim")))
            else:
                for ln in context_after[:2]:
                    parts.append(Text.assemble(("  ", ""), (ln, "dim")))
                hidden = len(context_after) - 3
                parts.append(Text.from_markup(
                    f"  [dim italic]\u2026 {hidden} more line{'s' if hidden != 1 else ''} \u2026[/dim italic]"
                ))
                parts.append(Text.assemble(("  ", ""), (context_after[-1], "dim")))
    else:
        # Fallback: no individual line matched (pattern spans lines)
        command_line = Text.assemble(
            ("$ ", "bold bright_black"),
            (command.strip(), "bold bright_white"),
        )
        parts.append(command_line)

    return Group(*parts)


def prompt_panel_confirmation(
    body,
    *,
    subtitle: str = "confirm to run",
    question: str = "Allow this command?",
) -> bool:
    """Show a safety panel with a y/N prompt.  Returns True if approved."""
    console.print()
    console.print(jarv_panel(body, title="safety", subtitle=subtitle, padding=(1, 2)))

    prompt = f"[bold]{question}[/bold] [dim]\\[y/N][/dim] [bold cyan]\u203a[/bold cyan] "
    try:
        choice = console.input(prompt).strip().lower()
    except EOFError:
        console.print("[dim]  denied.[/dim]")
        return False

    approved = choice in ("y", "yes")
    if approved:
        console.print("[green]  \u2713 approved[/green]\n")
    else:
        console.print("[red]  \u2717 denied[/red]\n")
    return approved


def approval_lock() -> threading.Lock:
    """Lock serializing interactive approval prompts across tool workers."""
    return _APPROVAL_LOCK


def prompt_confirmation(command: str, reason: str) -> bool:
    """Ask the user to approve a risky command.  Returns True if approved."""
    return prompt_panel_confirmation(_build_confirmation_body(command, reason))


def check_command(
    command: str,
    safety_level: str,
    audit: bool = False,
    config: dict | None = None,
    history: list | None = None,
    usage_path: Path | None = None,
    session_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> tuple[bool, str]:
    """Gate a command according to the configured safety level.

    Returns (allowed, denial_message).
    - allowed=True  → caller should execute the command.
    - allowed=False → caller should return denial_message to the model.

    When `audit=True`, flagged commands are sent to an LLM auditor. If the
    auditor approves, the command runs automatically. If it defers, the user
    gets the standard confirmation prompt with the auditor's reason.
    """
    if safety_level == "none":
        return True, ""

    with _APPROVAL_LOCK:
        return _check_command_locked(
            command, safety_level, audit, config, history,
            usage_path, session_id, cancellation_token,
        )


def _check_command_locked(
    command: str,
    safety_level: str,
    audit: bool,
    config: dict | None,
    history: list | None,
    usage_path: Path | None,
    session_id: str | None,
    cancellation_token: CancellationToken | None,
) -> tuple[bool, str]:
    if safety_level == "all":
        reason = "all commands require approval"
        if audit:
            return _audit_gate(
                command, reason, config, history, usage_path, session_id,
                cancellation_token,
            )
        if not prompt_confirmation(command, reason):
            return False, "[command denied by user — safety level is set to 'all']"
        return True, ""

    # "risky" (default)
    is_risky, reason = classify_command(command)
    if is_risky:
        if audit:
            return _audit_gate(
                command, reason, config, history, usage_path, session_id,
                cancellation_token,
            )
        if not prompt_confirmation(command, reason):
            return False, f"[command denied by user — detected as risky: {reason}]"
    return True, ""


_AUDITOR_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class _AuditPanel:
    """Live renderable: safety panel with auditor spinner + prompt line."""

    def __init__(self, body, start_time, audit_state, buf):
        self.body = body
        self.start = start_time
        self.audit_state = audit_state
        self.buf = buf
        self.show_auditor = True
        self.answered = False

    def __rich__(self):
        parts = [self.body]
        if self.show_auditor:
            parts.append(Text(""))
            if self.audit_state["done"]:
                reason = escape(self.audit_state["reason"])
                if self.audit_state["allow"]:
                    parts.append(Text.from_markup(
                        f"[green]✓  auditor[/green]  [dim]{reason}[/dim]"
                    ))
                else:
                    parts.append(Text.from_markup(
                        f"[yellow]⚠  auditor[/yellow]  [dim]{reason}[/dim]"
                    ))
            else:
                now = time.perf_counter()
                elapsed = int(now - self.start)
                frame = _AUDITOR_FRAMES[int(now * 10) % len(_AUDITOR_FRAMES)]
                parts.append(Text.from_markup(
                    f"[green]{frame}  auditor[/green]  [dim]checking…  {elapsed}s[/dim]"
                ))

        panel = jarv_panel(
            Group(*parts), title="safety", subtitle="confirm to run", padding=(1, 2),
        )
        buf_str = escape("".join(self.buf))
        if self.answered and buf_str:
            is_yes = buf_str.strip().lower() in ("y", "yes")
            color = "green" if is_yes else "red"
            prompt = Text.from_markup(
                f"[bold]Allow this command?[/bold] [dim]\\[y/N][/dim] [bold cyan]›[/bold cyan] [{color}]{buf_str}[/{color}]"
            )
        else:
            prompt = Text.from_markup(
                f"[bold]Allow this command?[/bold] [dim]\\[y/N][/dim] [bold cyan]›[/bold cyan] {buf_str}"
            )
        return Group(panel, prompt)


def _audit_gate(
    command: str,
    reason: str,
    config: dict | None,
    history: list | None,
    usage_path: Path | None,
    session_id: str | None,
    cancellation_token: CancellationToken | None,
) -> tuple[bool, str]:
    """Show the safety panel with an integrated auditor spinner.

    The auditor runs in a background thread while the user sees an animated
    spinner inside the panel.  The y/N prompt is rendered below the panel as
    part of the same Live block, so everything stays cohesive.
    """
    from .auditor import audit_command

    body = _build_confirmation_body(command, reason)

    # Non-interactive fallback
    if not sys.stdout.isatty():
        console.print()
        console.print(jarv_panel(body, title="safety", subtitle="confirm to run", padding=(1, 2)))
        allow, auditor_reason = audit_command(
            command,
            reason,
            config or {},
            history,
            usage_path=usage_path,
            session_id=session_id,
            cancellation_token=cancellation_token,
        )
        if allow:
            console.print(f"[green]  ✓ auditor:[/green] [dim]{auditor_reason}[/dim]")
            return True, ""
        return False, f"[command denied — auditor: {auditor_reason}]"

    audit_state: dict = {"done": False, "allow": False, "reason": ""}

    def _run_auditor():
        try:
            a, r = audit_command(
                command,
                reason,
                config or {},
                history,
                usage_path=usage_path,
                session_id=session_id,
                cancellation_token=cancellation_token,
            )
            audit_state["allow"] = a
            audit_state["reason"] = r
        except TurnCancelled:
            audit_state["cancelled"] = True
            audit_state["reason"] = "cancelled"
        except Exception as e:
            audit_state["reason"] = f"auditor unavailable ({type(e).__name__})"
        audit_state["done"] = True

    thread = threading.Thread(target=_run_auditor, daemon=True)
    thread.start()

    auto_approve = (config or {}).get("auditor_auto_approve", True)
    if live_display_depth() > 0:
        approved = _audit_poll_without_live(
            body,
            audit_state,
            auto_approve=auto_approve,
            cancellation_token=cancellation_token,
        )
    else:
        approved = _live_audit_poll(
            body,
            audit_state,
            auto_approve=auto_approve,
            cancellation_token=cancellation_token,
        )

    if approved:
        return True, ""
    deny = (
        audit_state["reason"]
        if audit_state.get("done") and not audit_state["allow"]
        else reason
    )
    return False, f"[command denied by user — detected as risky: {deny}]"


# ---------------------------------------------------------------------------
# Live panel + concurrent keyboard polling
# ---------------------------------------------------------------------------

def _audit_poll_without_live(
    body,
    audit_state: dict,
    *,
    auto_approve: bool = True,
    cancellation_token: CancellationToken | None = None,
) -> bool:
    """Wait for the auditor and prompt without starting a nested Rich Live."""
    console.print()
    console.print(jarv_panel(body, title="safety", subtitle="confirm to run", padding=(1, 2)))
    while not audit_state.get("done"):
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        if audit_state.get("cancelled"):
            raise TurnCancelled
        time.sleep(0.05)

    if auto_approve and audit_state["allow"]:
        console.print(f"[green]  ✓ auditor:[/green] [dim]{audit_state['reason']}[/dim]")
        console.print("[green]  ✓ approved[/green]\n")
        return True

    if not audit_state["allow"]:
        console.print(f"[yellow]  ⚠ auditor:[/yellow] [dim]{audit_state['reason']}[/dim]")
    console.print(
        "[bold]Allow this command?[/bold] [dim]\\[y/N][/dim] [bold cyan]›[/bold cyan] ",
        end="",
    )
    try:
        choice = console.input("").strip().lower()
    except EOFError:
        console.print("[dim]  denied.[/dim]")
        return False
    approved = choice in ("y", "yes")
    if approved:
        console.print("[green]  ✓ approved[/green]\n")
    else:
        console.print("[red]  ✗ denied[/red]\n")
    return approved


def _read_key():
    """Non-blocking single-character read.  Returns the char or None."""
    if sys.platform == "win32":
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch()
        return None
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _live_audit_poll(
    body,
    audit_state: dict,
    *,
    auto_approve: bool = True,
    cancellation_token: CancellationToken | None = None,
) -> bool:
    """Render the safety panel + auditor + prompt with Rich Live while
    polling for keyboard input and auditor completion concurrently."""
    from rich.live import Live

    start = time.perf_counter()
    buf: list[str] = []
    renderable = _AuditPanel(body, start, audit_state, buf)

    # Unix: switch to character-at-a-time input
    restore_term = None
    if sys.platform != "win32":
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        restore_term = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)

    approved = False

    try:
        console.print()
        with Live(renderable, refresh_per_second=10, console=console) as live:
            console.show_cursor(True)
            prefilled = False
            while True:
                if cancellation_token is not None:
                    cancellation_token.throw_if_cancelled()
                if audit_state.get("cancelled"):
                    raise TurnCancelled
                # Pre-fill the auditor's recommendation once it arrives, but
                # only if the user hasn't already started typing.
                if audit_state["done"] and not buf and not prefilled:
                    buf.append("y" if audit_state["allow"] else "n")
                    prefilled = True
                    live.refresh()

                # Auto-approve: skip the prompt entirely when enabled and the
                # auditor has finished with an allow decision.
                if auto_approve and audit_state["done"] and audit_state["allow"] and prefilled:
                    renderable.answered = True
                    live.refresh()
                    approved = True
                    break

                ch = _read_key()
                if ch is not None:
                    if ch in ("\r", "\n"):
                        if not audit_state["done"]:
                            renderable.show_auditor = False
                        renderable.answered = True
                        live.refresh()
                        choice = "".join(buf).strip().lower()
                        approved = choice in ("y", "yes")
                        break
                    elif ch == "\x03":
                        if not audit_state["done"]:
                            renderable.show_auditor = False
                        buf.clear()
                        live.refresh()
                        raise KeyboardInterrupt
                    elif ch in ("\x08", "\x7f"):
                        if buf:
                            buf.pop()
                    elif ch >= " ":
                        buf.append(ch)

                time.sleep(0.05)
    finally:
        if restore_term:
            restore_term()

    if not approved:
        console.print()
    return approved
