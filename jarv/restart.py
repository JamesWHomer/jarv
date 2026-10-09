"""Relaunch heads-up mode after its terminal and clients have been closed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


_PYTHON_RESTART_BOOTSTRAP = (
    "import sys; sys.path[0] = sys.argv.pop(1); from jarv.cli import main; main()"
)


def _restart_command(args: argparse.Namespace | None, session_id: str) -> list[str]:
    """Rebuild runtime options without replaying a prompt or ``--new``."""
    command = [sys.executable]
    if not getattr(sys, "frozen", False):
        # Preserve this installation's import origin after --cwd, including
        # when the target project contains an unrelated package named jarv.
        # The bootstrap changes only the child's Python path, not PYTHONPATH
        # inherited by its future tool subprocesses.
        source_root = str(Path(__file__).resolve().parent.parent)
        command += ["-c", _PYTHON_RESTART_BOOTSTRAP, source_root]

    for name in (
        "provider", "model", "effort", "timeout", "base_url", "service_tier",
        "command_safety", "max_turns", "run_timeout",
    ):
        value = getattr(args, name, None)
        if value is not None:
            command.append(f"--{name.replace('_', '-')}={value}")
    for key, value in getattr(args, "config", None) or []:
        encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        command.append(f"--config={key}={encoded}")

    system_file = getattr(args, "system_file", None)
    if system_file is not None:
        # main records this before applying --cwd; resolve the original file
        # from there, even when restarting from the requested project directory.
        base = getattr(args, "_restart_invocation_cwd", None)
        if base is None and hasattr(args, "system_file_text"):
            command.append(f"--system={args.system_file_text}")
        else:
            path = Path(system_file).expanduser()
            if not path.is_absolute():
                path = Path(base or os.getcwd()) / path
            command.append(f"--system-file={path.resolve()}")
    elif getattr(args, "system", None) is not None:
        command.append(f"--system={args.system}")

    if getattr(args, "cwd", None):
        # The process is already in the effective directory. Reapplying the
        # original relative --cwd would descend into that directory twice.
        command.append(f"--cwd={os.getcwd()}")
    if getattr(args, "incognito", False):
        command.append("--incognito")
    elif getattr(args, "session", None) is not None:
        # /new and /session can change a named invocation's current session.
        command.append(f"--session={session_id}")
    tools = getattr(args, "tools", None)
    if tools is not None:
        command.append(f"--tools={','.join(tools)}")
    for name in ("no_tools", "no_project_context", "no_update_check", "no_color"):
        if getattr(args, name, False):
            command.append(f"--{name.replace('_', '-')}")
    return command


def restart_heads_up(args: argparse.Namespace | None, session_id: str) -> int:
    """Replace jarv, or wait for its replacement on Windows.

    Windows needs a waiting parent so the invoking shell does not resume and
    compete with the new heads-up UI for input. Repeated restarts therefore
    retain idle waiting parents until the final child exits.
    """
    from .history import _RESTART_TERMINAL_ENV, detect_terminal

    command = _restart_command(args, session_id)
    environment = os.environ.copy()
    # A subprocess changes the fallback terminal identity based on parent PID.
    # Keep normal terminal/session binding rather than forcing a named session.
    environment[_RESTART_TERMINAL_ENV] = json.dumps(detect_terminal())
    if getattr(sys, "frozen", False):
        # PyInstaller must unpack a fresh application instance, including after
        # an installed binary has been updated in place.
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"

    # exec does not run Python finalizers. Do not import shell machinery merely
    # to restart an app that has not run any tools yet.
    shell_module = sys.modules.get("jarv.shell")
    state = getattr(shell_module, "_session_shell_state", None)
    if state is not None:
        state.close()

    if os.name != "nt":
        os.execve(command[0], command, environment)
        return 0  # Only reached by test doubles; a successful exec never returns.

    child = subprocess.Popen(command, env=environment)
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            # Windows broadcasts Ctrl+C to attached console processes. The new
            # heads-up app owns its meaning; its waiting parent must stay alive.
            continue
