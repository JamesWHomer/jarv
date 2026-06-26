"""Interactive ``/tree`` view: navigate the session prompt-tree and act on it.

Renders the tree built by :mod:`jarv.session_tree` as a navigable menu on the
shared :class:`~jarv.tui_app.AltScreenApp` loop (same chrome as the session
browser). Selecting a prompt can **resume** from it, **fork** a fresh line, or
**edit** it -- those compute a :class:`TreeOutcome` that the caller applies via
:func:`jarv.session_tree.checkout`. **Pruning** a branch is the one in-view edit:
it deletes straight away (with a two-press confirm) and reloads the tree in place.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from .display import console, terminal_size
from .history import branches_file_for, load_branches, load_history
from .session_tree import build_tree, delete_subtree, leaf_of, parent_id_of
from .tui_app import AltScreenApp
from .tui_frame import panel_width, wrap_frame
from .tui_layout import append_bottom_footer, clip_text
from .tui_overlay import body_content_rows, clamp_scroll_offset

@dataclass
class TreeOutcome:
    """What the user chose in the tree view; applied by the caller."""

    action: str            # "open" | "fork" | "edit" | "cancel"
    leaf_id: str | None    # frame id to check out ("" = root) — None for cancel
    prefill: str | None    # prompt text to pre-fill the editor (edit only)


class TreeBrowserScreen(AltScreenApp):
    text_mode = False
    use_mouse_capture = False
    use_bracketed_paste = False
    clear_on_resize = False
    first_paint_label = "tree"

    def __init__(self, *, model, history_file=None):
        super().__init__(
            console=console,
            repeatable_keys=frozenset({"UP", "DOWN", "PAGEUP", "PAGEDOWN"}),
        )
        self.model = model
        self.history_file = history_file
        self.nodes = model.nodes
        self.outcome = TreeOutcome("cancel", None, None)
        self.offset = 0
        self.arm_delete_id: str | None = None
        self.flash: tuple[str, str] | None = None  # (message, style) shown above the footer
        self.index_by_id = {id(n): i for i, n in enumerate(self.nodes)}
        self.connectors = self._compute_connectors()
        self.selected = next(
            (i for i, n in enumerate(self.nodes) if n.is_active_leaf),
            0,
        )

    # ------------------------------------------------------------------ #
    # One-time tree art (├─ └─ │) for each node, by depth and last-child.
    # ------------------------------------------------------------------ #
    def _compute_connectors(self) -> dict[int, str]:
        conn: dict[int, str] = {}

        def walk(node, trail: list[bool]) -> None:
            if not trail:
                conn[id(node)] = ""
            else:
                segs = ["   " if is_last else "│  " for is_last in trail[:-1]]
                segs.append("└─ " if trail[-1] else "├─ ")
                conn[id(node)] = "".join(segs)
            kids = node.children
            for idx, child in enumerate(kids):
                walk(child, trail + [idx == len(kids) - 1])

        for root in self.model.roots:
            walk(root, [])
        return conn

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self) -> Panel:
        term_w, term_h = terminal_size(console=console)
        width = panel_width(term_w)
        inner = max(1, width - 4)
        body_rows, show_footer = body_content_rows(term_h, footer_rows=3)

        if not self.nodes:
            parts: list = [
                Text(""),
                Text(
                    "  No prompts yet — start a conversation, then /tree to branch it.",
                    style="dim",
                ),
            ]
            if show_footer:
                append_bottom_footer(parts, term_h, self._footer_line("esc close", inner))
            return self._panel(parts, width, term_h)

        self.offset = self._clamp(body_rows)
        start = self.offset
        end = min(len(self.nodes), start + body_rows)
        parts = [self._row(self.nodes[i], i == self.selected, inner) for i in range(start, end)]

        if show_footer:
            selected_node = self.nodes[self.selected]
            if self.flash is not None:
                msg, style = self.flash
                status_line = Text("  " + clip_text(msg, inner - 2), style=style, no_wrap=True, overflow="ellipsis")
            else:
                status = selected_node.response_preview or "(no response yet)"
                status_line = Text("  ↳ " + clip_text(status, inner - 4), style="dim", no_wrap=True, overflow="ellipsis")
            append_bottom_footer(
                parts,
                term_h,
                Group(status_line, self._footer_line(self._footer_text(selected_node), inner)),
                footer_rows=3,
            )
        return self._panel(parts, width, term_h)

    def _row(self, node, selected: bool, inner: int) -> Text:
        conn = self.connectors.get(id(node), "")
        glyph = "● " if node.is_active_leaf else ""
        tag = f"  ⑂{len(node.children)}" if len(node.children) > 1 else ""
        used = 3 + len(conn) + len(glyph) + len(tag)
        label = clip_text(node.prompt_text or "(no prompt)", max(4, inner - used))

        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(" › " if selected else "   ", style="bold cyan" if selected else "")
        line.append(conn, style="bright_black")
        if glyph:
            line.append(glyph, style="cyan")
        if selected:
            label_style = "bold bright_white"
        elif node.on_active_path:
            label_style = "cyan"
        else:
            label_style = "dim"
        line.append(label, style=label_style)
        if tag:
            line.append(tag, style="dim")
        return line

    def _footer_text(self, node) -> str:
        # Edit re-sends a prompt as a sibling; only offer it on a leaf, where
        # there is no continuation to displace. Delete only applies to branches —
        # the active path is the live conversation. On a parent, Enter only dives
        # to the leaf; resume is reserved for the leaf itself.
        enter_hint = "enter ↓ leaf" if node.children else "enter resume"
        hints = ["↑↓ navigate", "←→ parent/child", enter_hint, "f fork"]
        if not node.children:
            hints.append("e edit")
        if not node.on_active_path:
            hints.append("d delete")
        hints.append("esc close")
        return " · ".join(hints)

    def _footer_line(self, text: str, inner: int) -> Text:
        return Text("  " + clip_text(text, inner - 2), style="dim italic", no_wrap=True, overflow="crop")

    def _subtitle(self) -> str:
        return f"[dim]{len(self.model.active_path)} on path · {len(self.nodes)} prompts[/dim]"

    def _panel(self, parts: list, width: int, term_h: int):
        # wrap_frame clears a previous, wider frame's stale right border on
        # WSL/ConPTY -- the same fix heads-up uses, applied here too.
        return wrap_frame(
            Panel(
                Group(*parts),
                title="[bold bright_white]jarv ▸ tree[/bold bright_white]",
                title_align="left",
                subtitle=self._subtitle(),
                subtitle_align="right",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
                width=width,
                height=term_h,
            )
        )

    def _clamp(self, body_rows: int) -> int:
        off = self.offset
        if self.selected < off:
            off = self.selected
        elif self.selected >= off + body_rows:
            off = self.selected - body_rows + 1
        return clamp_scroll_offset(off, len(self.nodes), body_rows)

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #
    def on_interrupt(self) -> None:
        self.stop()

    def _page(self) -> int:
        _, term_h = terminal_size(console=console)
        return max(1, body_content_rows(term_h, footer_rows=3)[0] - 1)

    def on_key(self, key: str, repeat: int) -> None:
        n = len(self.nodes)
        if n == 0:
            if key in ("ESC", "q", "Q"):
                self.stop()
            return

        # A pending delete is a one-key arm; Esc or any non-delete key cancels it.
        if key == "ESC" and self.arm_delete_id is not None:
            self.arm_delete_id = None
            self.flash = None
            return
        if key not in ("d", "D"):
            self.arm_delete_id = None
            self.flash = None

        node = self.nodes[self.selected]
        if key == "UP":
            self.selected = max(0, self.selected - repeat)
        elif key == "DOWN":
            self.selected = min(n - 1, self.selected + repeat)
        elif key == "LEFT":
            if node.parent is not None:
                self.selected = self.index_by_id[id(node.parent)]
        elif key == "RIGHT":
            if node.children:
                self.selected = self.index_by_id[id(node.children[0])]
        elif key == "HOME":
            self.selected = 0
        elif key == "END":
            self.selected = n - 1
        elif key == "PAGEUP":
            self.selected = max(0, self.selected - self._page() * repeat)
        elif key == "PAGEDOWN":
            self.selected = min(n - 1, self.selected + self._page() * repeat)
        elif key in ("ENTER", "o", "O"):
            if node.children:
                # Enter on a parent used to silently resume its leaf, which read
                # as a surprise fork. Instead just dive the cursor to that leaf
                # so resume/fork stay deliberate acts on the leaf itself.
                self.selected = self.index_by_id[id(leaf_of(node))]
            else:
                self.outcome = TreeOutcome("open", node.frame_id, None)
                self.stop()
        elif key in ("f", "F"):
            self.outcome = TreeOutcome("fork", node.frame_id, None)
            self.stop()
        elif key in ("e", "E"):
            if not node.children:  # edit only on leaves
                self.outcome = TreeOutcome("edit", parent_id_of(node), node.prompt_text)
                self.stop()
        elif key in ("d", "D"):
            self._on_delete(node)
        elif key in ("ESC", "q", "Q"):
            self.stop()

    def _on_delete(self, node) -> None:
        if node.on_active_path:
            self.arm_delete_id = None
            self.flash = ("Can't delete the active branch — it's the live thread.", "bold yellow")
            return
        if self.arm_delete_id == node.frame_id:
            select_index = (
                self.index_by_id[id(node.parent)] if node.parent is not None
                else max(0, self.selected - 1)
            )
            if self.history_file is not None:
                delete_subtree(self.history_file, node_id=node.frame_id)
            self.arm_delete_id = None
            self._reload(select_index=select_index)
            self.flash = ("Branch deleted.", "green")
            return
        self.arm_delete_id = node.frame_id
        below = " and everything below it" if node.children else ""
        self.flash = (f"Delete this branch{below}? Press d again to confirm · any key cancels.", "bold red")

    def _reload(self, *, select_index: int | None = None) -> None:
        self.model = build_tree(
            load_history(self.history_file),
            load_branches(branches_file_for(self.history_file)),
        )
        self.nodes = self.model.nodes
        self.index_by_id = {id(n): i for i, n in enumerate(self.nodes)}
        self.connectors = self._compute_connectors()
        if not self.nodes:
            self.selected = 0
        elif select_index is not None:
            self.selected = max(0, min(select_index, len(self.nodes) - 1))
        else:
            self.selected = min(self.selected, len(self.nodes) - 1)
        self.offset = 0


def run_tree_screen(session_context, config=None) -> TreeOutcome:
    """Build the tree for the active session and run the interactive view."""
    history = load_history(session_context.history_file)
    branches = load_branches(branches_file_for(session_context.history_file))
    model = build_tree(history, branches)
    screen = TreeBrowserScreen(model=model, history_file=session_context.history_file)
    screen.run()
    return screen.outcome
