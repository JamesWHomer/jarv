import sys

from .display import console


SETUP_STEPS = {"provider", "key", "model", "base_url"}


def run_setup_wizard(step: str | None = None) -> dict | None:
    """Run the interactive setup wizard.

    A thin entry point onto the fullscreen wizard (:func:`run_setup_interactive`),
    which owns the whole experience -- the animated welcome, the per-setting
    submenus ``/settings`` uses, the background connection check, and the ready
    summary. There is no inline console chrome around it.

    If *step* is given, only that setting is opened (provider, key, model,
    base_url). Returns the updated config on success, ``None`` if cancelled.
    """
    from .config import load_config

    config = load_config()

    if not sys.stdin.isatty() or not console.is_terminal:
        console.print(
            "[yellow]Run [bold]jarv /setup[/bold] in an interactive terminal to configure jarv.[/yellow]"
        )
        return None

    from .setup_interactive import run_setup_interactive

    return run_setup_interactive(config, step=step)
