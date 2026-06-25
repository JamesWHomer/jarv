"""Standalone ``/tree`` entry point (used outside the heads-up loop).

The polished in-app flow lives in ``headsup._run_tree`` (it can pre-fill the
editor and re-sync the transcript). This handler covers ``jarv /tree`` from a
plain shell: it runs the same interactive view and applies the chosen action to
disk, so the *next* invocation continues from the new active head.
"""

from __future__ import annotations

import sys

from .display import console
from .history import branches_file_for, load_branches, load_history, prepare_session_context


def cmd_tree(args: list | None = None) -> None:
    ctx = prepare_session_context()
    if not sys.stdin.isatty() or not console.is_terminal:
        _tree_plain(ctx)
        return

    from .commands import load_config
    from .tree_browser import run_tree_screen

    outcome = run_tree_screen(ctx, load_config())
    _apply_standalone(ctx, outcome)


def _apply_standalone(ctx, outcome) -> None:
    from . import session_tree

    if outcome.action == "cancel":
        console.print("[dim]○ Closed tree.[/dim]")
        return

    changed = session_tree.checkout(ctx.history_file, leaf_id=outcome.leaf_id)
    if outcome.action == "open":
        if changed:
            console.print("[bold green]✓[/bold green] [green]Resumed from the selected prompt.[/green]")
        else:
            console.print("[dim]○ Already on that prompt.[/dim]")
    elif outcome.action == "fork":
        console.print(
            "[bold cyan]⑂[/bold cyan] [cyan]Forked.[/cyan] "
            "[dim]Your next message starts a new branch.[/dim]"
        )
    elif outcome.action == "edit":
        console.print(
            "[bold cyan]✎[/bold cyan] [cyan]Ready to edit.[/cyan] "
            "[dim]Re-send your revised prompt:[/dim]"
        )
        if outcome.prefill:
            console.print(f"  [dim]{outcome.prefill}[/dim]")


def _tree_plain(ctx) -> None:
    """Non-interactive fallback: print the tree as indented text."""
    from rich.console import Group
    from rich.text import Text

    from .display import jarv_panel
    from .session_tree import build_tree

    model = build_tree(
        load_history(ctx.history_file),
        load_branches(branches_file_for(ctx.history_file)),
    )
    if not model.nodes:
        console.print("[yellow]No prompts yet in this session.[/yellow]")
        return

    lines: list = []
    for node in model.nodes:
        indent = "  " * node.depth
        marker = "● " if node.is_active_leaf else ""
        text = (node.prompt_text or "(no prompt)").replace("\n", " ")
        lines.append(
            Text(
                f"{indent}{marker}{text[:80]}",
                style="cyan" if node.on_active_path else "white",
            )
        )
    console.print(jarv_panel(Group(*lines), title="tree", subtitle=f"{len(model.nodes)} prompts"))
    console.print("[dim italic]Run /tree in an interactive terminal to fork, edit, or resume.[/dim italic]")
