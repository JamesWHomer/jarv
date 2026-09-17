"""Lightweight status labels shared by the agent and the heads-up menu."""

from rich.text import Text

STREAM_PREVIEW_REFRESH_INTERVAL = 1 / 12

_TOOL_ACTIVITY_LABELS = {
    "run_command": ("Writing command", "Wrote command"),
    "spawn": ("Planning parallel tasks", "Planned parallel tasks"),
    "read": ("Selecting content", "Selected content"),
    "edit": ("Writing edit", "Wrote edit"),
    "ask_user": ("Writing question", "Wrote question"),
    "web_search": ("Writing web search", "Wrote web search"),
}


def tool_activity_label(tool_names: tuple[str, ...]) -> str:
    """Return the live activity label for tool-call serialization."""
    if len(tool_names) != 1:
        return f"Preparing {len(tool_names)} actions"
    return _TOOL_ACTIVITY_LABELS.get(
        tool_names[0], ("Preparing action", "Prepared action"),
    )[0]


def thought_complete_indicator(text: str) -> Text:
    """Return the static completed-thinking bubble."""
    return Text(f"\u2726 {text}", style="dim")


def tool_complete_indicator(text: str) -> Text:
    """Return the static completed-tool bubble."""
    return Text(f"\u2713 {text}", style="dim")
