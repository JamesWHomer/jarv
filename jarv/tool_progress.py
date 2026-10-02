"""Live, per-call progress for parallel reads and web searches."""

import json
import time
from contextlib import ExitStack

from rich.console import Group
from rich.live import Live
from rich.text import Text

from .cancellation import TurnCancelled
from .config import get_setting
from .display import console, track_live_display
from .response_wait import _THINKING_FRAMES
from .session_render import ToolCallCard, _tool_call_renderable
from .tool_outputs import ToolOutput, summarize_tool_output, tool_outcome, with_tool_outcome


class LiveToolCallCard(ToolCallCard):
    def __init__(
        self, item: dict, output: str, display_mode: str, *,
        state: str, started_at: float | None = None,
    ):
        super().__init__(item, output, display_mode)
        self.live_key = "tool:" + str(item["call_id"])
        self.state = state
        self.started_at = started_at
        self.finished = state == "finished"

    def __rich_console__(self, console, options):
        status = None
        if self.state == "queued":
            status = ("queued", "yellow")
        elif self.state == "running":
            now = time.perf_counter()
            elapsed = int(max(0, now - self.started_at))
            frame = _THINKING_FRAMES[int(now * 10) % len(_THINKING_FRAMES)]
            status = (f"{frame} running {elapsed}s", "blue")
        else:
            outcome = tool_outcome(self.output)
            if outcome is not None and outcome.status in {"cancelled", "timed_out", "denied", "skipped"}:
                status = (outcome.status.replace("_", " "), "red")
        yield _tool_call_renderable(
            self.item, self.output, display_mode=self.display_mode,
            expanded=self.expanded, status_override=status,
        )


class ParallelToolDisplay:
    """One terminal Live, or independent heads-up slots, for an entire batch.

    Updates run on the execution coordinator, never on pool workers. Each
    update replaces a card so the renderer sees a consistent snapshot.
    """

    def __init__(self, config: dict, ui=None):
        self.config = config
        self.ui = ui
        self.cards: dict[str, LiveToolCallCard] = {}
        self.live: Live | None = None
        self._stack = ExitStack()
        self.live_updates = (
            console.is_terminal if ui is None
            else getattr(ui, "supports_live_tool_cards", True)
        )

    def update(self, item, args: dict | None, state: str, output: ToolOutput) -> None:
        previous = self.cards.get("tool:" + str(item.call_id))
        started_at = previous.started_at if previous is not None else None
        if state == "running" and started_at is None:
            started_at = time.perf_counter()
        card = LiveToolCallCard(
            {"name": item.name, "call_id": item.call_id,
             "arguments": json.dumps(args, ensure_ascii=True) if args is not None else item.arguments},
            summarize_tool_output(output), get_setting(self.config, "tool_call_display"),
            state=state, started_at=started_at,
        )
        self._show(card)

    def _show(self, card: LiveToolCallCard) -> None:
        self.cards[card.live_key] = card
        if self.config.get("_quiet") or not self.live_updates:
            return
        if self.ui is not None:
            handler = getattr(self.ui, "show_tool_card", None)
            if handler is not None:
                handler(card)
            return
        renderables = []
        for current in self.cards.values():
            renderables.append(current)
            if get_setting(self.config, "tool_call_display") == "print":
                renderables.append(Text(""))
        group = Group(*renderables)
        if self.live is None:
            self._stack.enter_context(track_live_display())
            self.live = self._stack.enter_context(Live(
                group, console=console, refresh_per_second=8,
                transient=False, vertical_overflow="crop",
            ))
        else:
            self.live.update(group, refresh=True)

    def finish(self, error: BaseException | None = None) -> None:
        try:
            if error is not None:
                cancelled = isinstance(error, (TurnCancelled, KeyboardInterrupt))
                for card in tuple(self.cards.values()):
                    if card.finished:
                        continue
                    status = "cancelled" if cancelled else "failed"
                    message = "cancelled" if cancelled else "interrupted by error"
                    self._show(LiveToolCallCard(
                        card.item, with_tool_outcome(f"[tool {message}]", status),
                        card.display_mode, state="finished",
                    ))
            # Pipes and protocol diagnostics cannot replace earlier rows.
            # Print final cards once, in the original call order.
            if not self.live_updates and not self.config.get("_quiet"):
                for card in self.cards.values():
                    if self.ui is not None:
                        self.ui.show_tool_card(card)
                    else:
                        console.print(card)
                        if get_setting(self.config, "tool_call_display") == "print":
                            console.print()
        finally:
            self._stack.close()
            self.live = None
            self.cards.clear()
