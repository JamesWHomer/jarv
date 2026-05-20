"""Undo and redo command handlers."""

from .display import console
from .history import (
    load_history,
    load_redo_stack,
    prepare_session_context,
    redo_file_for,
    save_history,
    save_redo_stack,
    split_last_exchange,
)


def _parse_count(args: list, default: int = 1) -> int:
    if not args:
        return default
    try:
        return max(1, int(args[0]))
    except ValueError:
        return default


def _first_user_text(frame: list) -> str:
    for item in frame:
        if isinstance(item, dict) and item.get("role") == "user":
            return str(item.get("content", "")).strip().replace("\n", " ")[:80]
    return "(no user message)"


def cmd_undo(args: list) -> None:
    n = _parse_count(args)
    ctx = prepare_session_context()
    history = load_history(ctx.history_file)
    redo_path = redo_file_for(ctx.history_file)
    stack = load_redo_stack(redo_path)

    undone: list[list] = []
    for _ in range(n):
        history, frame = split_last_exchange(history)
        if not frame:
            break
        undone.append(frame)
        stack.append(frame)

    if not undone:
        console.print("[dim]○ Nothing to undo.[/dim]")
        return

    save_history(history, ctx.history_file)
    save_redo_stack(stack, redo_path)

    if len(undone) == 1:
        text = _first_user_text(undone[0])
        console.print(f"[bold yellow]↶[/bold yellow] [bold]Unsent[/bold] [cyan]{text!r}[/cyan]")
        console.print(f"[dim]  Removed {len(undone[0])} item(s). Run [bold]/redo[/bold] to put it back.[/dim]")
    else:
        console.print(f"[bold yellow]↶[/bold yellow] [bold]Unsent {len(undone)} exchanges:[/bold]")
        for i, frame in enumerate(undone, 1):
            console.print(f"  [dim]{i}.[/dim] [cyan]{_first_user_text(frame)!r}[/cyan]")
        console.print(f"[dim]  Run [bold]/redo {len(undone)}[/bold] to put them back.[/dim]")


def cmd_redo(args: list) -> None:
    n = _parse_count(args)
    ctx = prepare_session_context()
    history = load_history(ctx.history_file)
    redo_path = redo_file_for(ctx.history_file)
    stack = load_redo_stack(redo_path)

    restored: list[list] = []
    for _ in range(n):
        if not stack:
            break
        frame = stack.pop()
        history.extend(frame)
        restored.append(frame)

    if not restored:
        console.print("[dim]○ Nothing to redo.[/dim]")
        return

    save_history(history, ctx.history_file)
    save_redo_stack(stack, redo_path)

    if len(restored) == 1:
        text = _first_user_text(restored[0])
        console.print(f"[bold cyan]↷[/bold cyan] [bold]Restored[/bold] [cyan]{text!r}[/cyan]")
    else:
        console.print(f"[bold cyan]↷[/bold cyan] [bold]Restored {len(restored)} exchange(s).[/bold]")

