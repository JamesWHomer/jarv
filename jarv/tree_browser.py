"""Interactive prompt-tree browser with folding, search, and exchange previews."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from .command_input import TextInput
from .display import console, terminal_size
from .session_browser_render import (
    beside, fitted, highlighted, highlighted_transcript, one_line, pane_heading, reflow_position,
)
from .session_render import _history_visual_lines
from .session_tree import delete_subtree, leaf_of, load_session_tree, parent_id_of
from .storage import StorageError
from .terminal_text import safe_terminal_text
from .text_editor import apply_text_editor_key, initialize_text_editor, render_single_line
from .tool_outputs import flatten_content_text
from .tree_browser_view import TreeRow, TreeView
from .tui_app import AltScreenApp
from .tui_frame import panel_width, wrap_frame
from .tui_panel import MenuPanel, menu_frame_rows, menu_inner_width
from .tui_overlay import (
    apply_scroll_keys, apply_selection_keys, clamp_scroll_offset,
    clamp_selection_scroll, scroll_position_hint,
)


@dataclass
class TreeOutcome:
    """What the user chose; the caller applies checkout after this view closes."""

    action: str            # "open" | "fork" | "edit" | "cancel"
    leaf_id: str | None    # frame id to check out ("" = root) — None for cancel
    prefill: str | None    # original prompt to pre-fill the editor (edit only)


class TreeBrowserScreen(AltScreenApp):
    use_mouse_capture = True
    use_bracketed_paste = True
    clear_on_resize = False
    first_paint_label = "tree"

    def __init__(self, *, model, history_file=None):
        super().__init__(
            console=console,
            repeatable_keys=frozenset({"UP", "DOWN", "PAGEUP", "PAGEDOWN"}),
        )
        self.model = model
        self.history_file = history_file
        self.outcome = TreeOutcome("cancel", None, None)
        self.offset = 0
        self.selected = next((i for i, node in enumerate(model.nodes) if node.is_active_leaf), 0)
        self.selected_run = False
        self.pane_focus = "tree"
        self.full_preview = False
        self.preview_return_focus = "tree"
        self.arm_delete_id: str | None = None
        self.flash: tuple[str, str] | None = None
        self.search_active = False
        self.search_query = ""
        self.search_editor: dict = {}
        self.search_matches: set[int] = set()
        self.search_included: set[int] | None = None
        self.search_text: list[str] | None = None
        self.help_open = False
        self.help_offset = 0
        self.preview_cache: OrderedDict[tuple, tuple[list[Text], int | None]] = OrderedDict()
        self.preview_positions: dict[tuple, tuple[int, int]] = {}
        self._reset_view()

    @property
    def text_mode(self) -> bool:
        # Keep q and shortcut-looking pasted text literal while editing a search.
        return self.search_active

    def _reset_view(self) -> None:
        self.nodes = self.model.nodes
        self.view = TreeView(self.model)
        self.index_by_id = self.view.index_by_id
        self.connectors = {
            id(self.nodes[row.index]): row.connector
            for row in self.view.rows(set(range(len(self.nodes))))
        }
        self._rows_cache: list[TreeRow] | None = None

    def _rows(self) -> list[TreeRow]:
        if self._rows_cache is None:
            self._rows_cache = self.view.rows(self.search_included)
        return self._rows_cache

    def _selected_row(self) -> TreeRow | None:
        return next((row for row in self._rows()
                     if row.index == self.selected and bool(row.run) == self.selected_run), None)

    def _select_node(self, index: int) -> None:
        self.selected, self.selected_run = index, False
        # Structural navigation may leave a filtered path. Clear the filter so
        # the real parent/child is always the target, even when it didn't match.
        if self.search_included is not None and index not in self.search_included:
            self.search_query = ""
            self.search_included = None
            self.search_matches.clear()
        self.view.reveal(index)
        self._rows_cache = None

    def _select_row(self, row: TreeRow) -> None:
        if row.run:
            self.selected, self.selected_run = row.index, True
        else:
            self._select_node(row.index)

    # ------------------------------------------------------------------ #
    # Geometry and shared menu chrome
    # ------------------------------------------------------------------ #
    def _layout(self) -> dict:
        term_w, term_h = terminal_size(console=console)
        width = panel_width(term_w)
        inner = menu_inner_width(width)
        available = max(1, term_h - menu_frame_rows())
        footer_rows = 2 if available >= 9 else int(available >= 2)
        status_rows = int(available >= 6 or (self.flash is not None and available >= 3))
        search_rows = int(bool(self.search_active or self.search_query) and available >= 5)
        body = max(1, available - footer_rows - status_rows - search_rows)
        split = term_w >= 120 and body >= 9 and not self.full_preview
        compact_preview = min(6, body // 3) if not split and not self.full_preview and body >= 12 else 0
        tree_height = body - compact_preview
        left = min(96, (inner - 3) * 3 // 5) if split else inner
        return dict(width=width, height=term_h, inner=inner, body=body, split=split,
                    left=left, right=inner - left - 3 if split else inner,
                    tree_height=tree_height, compact_preview=compact_preview,
                    heading=int(tree_height >= 5), footer_rows=footer_rows,
                    status_rows=status_rows, search_rows=search_rows)

    def _pair(self, left: str, right: str, width: int, *, style="dim") -> Text:
        line = Text(left, style=style)
        if right and line.cell_len + Text(right).cell_len + 3 <= width:
            line.append(" " * (width - line.cell_len - Text(right).cell_len) + right, style="dim")
        return fitted(line, width)

    def _footer_lines(self, layout: dict) -> list[Text]:
        width = layout["inner"]
        row = self._selected_row()
        if self.arm_delete_id is not None:
            options = ("d confirm delete · esc cancel", "d confirm · esc cancel", "d / esc cancel", "d / esc", "esc")
            primary = next((hint for hint in options if Text(hint).cell_len <= width), "esc")
            navigation, extra = "Any other key cancels deletion", ""
        elif self.search_active:
            primary = "enter results · esc clear"
            navigation = "Type to search prompts and replies"
            extra = ""
        elif self.full_preview or self.pane_focus == "preview":
            primary = "p restore · tab tree · ? help · esc back" if self.full_preview else "tab tree · p expand · ? help · esc back"
            navigation, extra = "↑↓ scroll · pgup/pgdn page · home/end", "ctrl+f search"
        elif row is None:
            primary, navigation, extra = "ctrl+f search · ? help · esc close", "", ""
        elif row.run:
            primary, navigation, extra = "enter expand · ? help · esc close", "↑↓ navigate · ←→ parent/expand · space expand", ""
        else:
            node = self.nodes[row.index]
            actions = ["enter ↓ leaf" if node.children else "enter resume", "f fork"]
            if not node.children:
                actions.append("e edit")
            if not node.on_active_path:
                actions.append("d delete")
            primary = " · ".join(actions + ["? help", "esc close"])
            navigation, extra = "↑↓ navigate · ←→ parent/child · space fold", "tab preview · p expand"

        if Text(primary).cell_len > width:
            enter = "enter expand" if row and row.run else "enter ↓ leaf" if row and self.nodes[row.index].children else "enter resume"
            if self.full_preview or self.pane_focus == "preview":
                primary = "↑↓ scroll · p restore · esc back" if self.full_preview else "↑↓ scroll · tab tree · esc back"
            elif self.search_active:
                primary = "enter results · esc clear"
            else:
                primary = f"{enter} · ? help · esc close" if row else "? help · esc close"
        if Text(primary).cell_len > width:
            primary = "? help · esc back" if self.full_preview else "? help · esc close"
        if Text(primary).cell_len > width:
            primary = "esc"
        show_search = not self.search_active and self.pane_focus == "tree" and self.arm_delete_id is None and "ctrl+f" not in primary
        actions_line = self._pair(primary, "ctrl+f search" if show_search else "", width)
        if layout["footer_rows"] < 2:
            return [actions_line] if layout["footer_rows"] else []
        if Text(navigation).cell_len > width:
            navigation = "↑↓ move · ←→ branch · p preview" if self.pane_focus == "tree" else "↑↓ scroll · pgup/pgdn page"
        return [self._pair(navigation, extra, width), actions_line]

    def _subtitle(self) -> str:
        return f"[dim]{len(self.nodes)} prompts · {len(self.model.active_path)} on current path[/dim]"

    def _panel(self, parts: list, layout: dict, title="tree"):
        return wrap_frame(MenuPanel(
            Group(*parts), title=f"[bold bright_white]jarv ▸ {title}[/bold bright_white]",
            title_align="left", subtitle=self._subtitle(), subtitle_align="right",
            border_style="cyan", box=box.ROUNDED, padding=(0, 1),
            width=layout["width"], height=layout["height"],
        ))

    def _search_line(self, width: int) -> Text:
        line = Text("Search: ", style="bold cyan" if self.search_active else "dim")
        count = f"{len(self.search_matches)} matches" if self.search_query else "prompts and replies"
        room = max(1, width - line.cell_len - (len(count) + 2 if width >= 50 else 0))
        if self.search_active:
            line.append_text(render_single_line(self.search_editor, room, text_style="bold"))
        else:
            line.append_text(fitted(Text(self.search_query), room))
        if width >= 50:
            line.append(" " * max(2, width - line.cell_len - len(count)) + count, style="dim")
        return fitted(line, width)

    def render(self) -> Panel:
        layout = self._layout()
        if self.help_open:
            return self._render_help(layout)
        if not layout["split"] and not self.full_preview:
            self.pane_focus = "tree"
        if self.full_preview:
            body = self._preview_lines(layout["inner"], layout["body"])
        elif layout["split"]:
            body = beside(self._tree_lines(layout["left"], layout["body"]),
                          self._preview_lines(layout["right"], layout["body"]),
                          layout["left"], layout["right"], layout["body"])
        else:
            body = self._tree_lines(layout["inner"], layout["tree_height"])
            if layout["compact_preview"]:
                body += self._preview_lines(layout["inner"], layout["compact_preview"], compact=True)
        parts = [self._search_line(layout["inner"])] if layout["search_rows"] else []
        parts.extend(body)
        if layout["status_rows"]:
            parts.append(fitted(Text(self.flash[0], style=self.flash[1]) if self.flash else Text(""), layout["inner"]))
        parts.extend(self._footer_lines(layout))
        return self._panel(parts, layout, "tree · preview" if self.full_preview else "tree")

    # ------------------------------------------------------------------ #
    # Tree rows: fixed cursor gutter, independent live marker, cell-safe fit
    # ------------------------------------------------------------------ #
    def _tree_lines(self, width: int, height: int) -> list[Text]:
        rows = self._rows()
        heading = int(height >= 5)
        capacity = max(1, height - heading)
        position = next((i for i, row in enumerate(rows)
                         if row.index == self.selected and bool(row.run) == self.selected_run), 0)
        self.offset = clamp_selection_scroll(self.offset, position, len(rows), capacity)
        end = min(len(rows), self.offset + capacity)
        count = scroll_position_hint(self.offset, end, len(rows)) if len(rows) > capacity else ""
        parts = [pane_heading("TREE", width, active=self.pane_focus == "tree" and not self.search_active, count=count)] if heading else []
        if not rows:
            message = "No matching prompts or replies." if self.search_query else "No prompts yet — start a conversation, then /tree to branch it."
            parts.append(fitted(Text(message, style="dim"), width))
        else:
            for row in rows[self.offset:end]:
                selected = row.index == self.selected and bool(row.run) == self.selected_run
                parts.append(self._render_row(row, selected, width))
        parts.extend(Text("") for _ in range(max(0, height - len(parts))))
        return parts[:height]

    def _render_row(self, row: TreeRow, selected: bool, width: int) -> Text:
        node = self.nodes[row.index]
        focused = selected and self.pane_focus == "tree" and not self.search_active
        line = Text(style="on #17333b" if focused else "on #15252b" if selected else "")
        line.append("› " if selected else "  ", style="bold cyan" if focused else "dim cyan")
        tail = "● current" if node.is_active_leaf else f"{row.hidden} hidden" if row.hidden else ""
        # Leave room for the prompt before spending columns on ancestry.
        label_room = max(1, width - 2 - (Text(tail).cell_len + 2 if tail else 0))
        budget = min(max(0, width // 2), max(0, label_room - 12))
        connector = row.connector
        if len(connector) > budget:
            connector = "… " + connector[-(budget - 2):] if budget > 2 else "…"[:budget]
        line.append(connector, style="bright_black")
        prompt = safe_terminal_text(node.prompt_text or "(no prompt)")
        label = f"▸ {len(row.run)} earlier prompts · {prompt}" if row.run else ("▸ " if row.hidden else "") + prompt
        style = ("bold red" if self.arm_delete_id == node.frame_id else
                 "bold bright_white" if selected else "cyan" if node.on_active_path else "")
        line.append_text(fitted(highlighted(label, self.search_query, style), max(1, label_room - len(connector)), pad=True))
        if tail:
            line.append("  " + tail, style="green" if node.is_active_leaf else "dim")
        return fitted(line, width, pad=True)

    def _row(self, node, selected: bool, inner: int) -> Text:
        """Render an individual prompt, including callers outside the viewport."""
        index = self.index_by_id[id(node)]
        row = next((row for row in self._rows() if row.index == index),
                   TreeRow(index, self.connectors.get(id(node), "")))
        return self._render_row(row, selected, inner)

    # ------------------------------------------------------------------ #
    # Preview documents and reading position survive focus and width changes
    # ------------------------------------------------------------------ #
    def _preview_document(self, index: int, width: int) -> tuple[list[Text], int | None]:
        node = self.nodes[index]
        width = min(96, max(1, width))
        key = (node.frame_id, width, self.search_query)
        cached = self.preview_cache.get(key)
        if cached is not None:
            self.preview_cache.move_to_end(key)
            return cached
        lines = _history_visual_lines(node.items, width)
        has_reply = any(isinstance(item, dict) and (
            item.get("type") in ("function_call", "status") or
            (item.get("role") == "assistant" and flatten_content_text(item.get("content", "")).strip())
        ) for item in node.items)
        if not has_reply:
            lines += [Text(""), Text("jarv:", style="bold green"), Text("(no response yet)", style="dim")]
        cached = highlighted_transcript(lines or [Text("(empty exchange)", style="dim")], self.search_query)
        self.preview_cache[key] = cached
        while len(self.preview_cache) > 24:
            self.preview_cache.popitem(last=False)
        return cached

    def _preview_window(self, index: int, width: int, height: int) -> tuple[list[Text], int, int]:
        width = min(96, max(1, width))
        lines, match = self._preview_document(index, width)
        key = (self.nodes[index].frame_id, self.search_query)
        previous_width, start = self.preview_positions.get(key, (width, match or 0))
        if previous_width != width:
            previous, _ = self._preview_document(index, previous_width)
            start = reflow_position(previous, start, lines)
        start = clamp_scroll_offset(start, len(lines), max(1, height))
        self.preview_positions[key] = (width, start)
        return lines[start:start + height], start, len(lines)

    def _preview_lines(self, width: int, height: int, *, compact=False) -> list[Text]:
        heading = int(height >= 3)
        footer = int(height >= 4 and not compact)
        capacity = max(1, height - heading - footer)
        parts = [pane_heading("PREVIEW", width, active=self.pane_focus == "preview")] if heading else []
        row = self._selected_row()
        position = ""
        if row is None:
            lines = [Text("Select a prompt to preview its exchange.", style="dim")]
        elif row.run:
            lines = [Text(f"{len(row.run)} earlier prompts", style="bold"),
                     Text("In the tree, Enter or Space expands this group.", style="dim")]
            lines += [Text(safe_terminal_text(self.nodes[index].prompt_text)) for index in row.run[:capacity]]
        elif compact:
            document, match = self._preview_document(row.index, width)
            start = match if match is not None else next((i for i, line in enumerate(document) if line.plain == "jarv:"), 0)
            lines = document[start:start + capacity]
        else:
            lines, start, total = self._preview_window(row.index, width, capacity)
            position = scroll_position_hint(start, min(total, start + capacity), total)
        parts.extend(fitted(line, width) for line in lines[:capacity])
        parts.extend(Text("") for _ in range(max(0, height - footer - len(parts))))
        if footer:
            hint = "↑↓ scroll · tab tree" if self.pane_focus == "preview" else "tab preview · p expand"
            parts.append(self._pair(hint, position, width))
        return parts[:height]

    def _scroll_preview(self, key: str, repeat: int, layout: dict) -> None:
        row = self._selected_row()
        if row is None or row.run:
            return
        width = layout["inner"] if self.full_preview else layout["right"]
        height = max(1, layout["body"] - int(layout["body"] >= 3) - int(layout["body"] >= 4))
        _, start, total = self._preview_window(row.index, width, height)
        start = apply_scroll_keys(key, repeat, offset=start, total=total, body_rows=height)
        self.preview_positions[(self.nodes[row.index].frame_id, self.search_query)] = (min(96, width), start)

    def _open_preview(self) -> None:
        self.preview_return_focus = self.pane_focus
        self.full_preview = True
        self.pane_focus = "preview"

    def _close_preview(self) -> None:
        self.full_preview = False
        self.pane_focus = self.preview_return_focus if self._layout()["split"] else "tree"

    # ------------------------------------------------------------------ #
    # Search and help
    # ------------------------------------------------------------------ #
    def _update_search(self) -> None:
        self.search_query = one_line(self.search_query)
        query = self.search_query.casefold()
        if query and self.search_text is None:
            self.search_text = [one_line(" ".join(
                flatten_content_text(item.get("content", "")) for item in node.items
                if isinstance(item, dict) and item.get("role") in ("user", "assistant")
            )).casefold() for node in self.nodes]
        self.search_matches = {i for i, text in enumerate(self.search_text or []) if query in text} if query else set()
        self.search_included = self.view.with_ancestors(self.search_matches) if query else None
        self._rows_cache = None
        self.offset = 0
        if query:
            self.selected_run = False
            if self.search_matches and self.selected not in self.search_matches:
                self.selected = min(self.search_matches)
        elif self.nodes:
            self._select_node(self.selected)

    def _on_key_search(self, key: str, repeat: int) -> None:
        if not isinstance(key, TextInput) and key == "ESC":
            self.search_active = False
            self.search_query = ""
            self._update_search()
        elif not isinstance(key, TextInput) and key in ("ENTER", "DOWN", "TAB", "CTRL_F"):
            self.search_active = False
        elif apply_text_editor_key(self.search_editor, key, repeat):
            # Search is a single line even when pasted text contains newlines.
            self.search_editor["buffer"] = self.search_editor["buffer"][:1000]
            self.search_editor["cursor"] = min(self.search_editor["cursor"], 1000)
            self.search_query = self.search_editor["buffer"]
            self._update_search()

    def _help_lines(self, width: int) -> list[Text]:
        content = [
            "Navigate", "↑↓ / mouse wheel    Move through visible prompts and folded groups",
            "← / →    Parent / child; entering a hidden prompt reveals it",
            "PgUp/PgDn · Home/End    Page or jump through the focused pane",
            "Space    Expand a group, fold a straight run, or toggle an inactive branch",
            "The live endpoint and its parent stay visible; fork points stay outside run groups.",
            "", "Preview and search", "Tab    Focus preview / return to tree; opens full preview on narrow terminals",
            "p    Expand / restore preview · Esc or ← returns",
            "Ctrl+F    Search prompts and replies, keeping matching prompts' ancestors",
            "Enter / ↓    Leave search input and browse results · Esc clears the search",
            "", "Actions (tree pane)", "Enter    Jump to the tip on a parent; resume on a leaf; expand a folded group",
            "f    Fork from the selected prompt", "e    Edit a leaf, preserving its original prompt text",
            "d, then d    Delete an inactive branch and its descendants; any other key cancels",
            "", "?    Show / close shortcuts", "Esc / q    Return from preview, clear search, then close",
        ]
        result = []
        for value in content:
            style = "bold cyan" if value in ("Navigate", "Preview and search", "Actions (tree pane)") else ""
            result.extend(Text(value, style=style).wrap(self.console, max(1, width)))
        return result

    def _render_help(self, layout: dict):
        lines = self._help_lines(layout["inner"])
        height = max(1, layout["height"] - menu_frame_rows() - 1)
        self.help_offset = clamp_scroll_offset(self.help_offset, len(lines), height)
        parts = lines[self.help_offset:self.help_offset + height]
        parts.extend(Text("") for _ in range(max(0, height - len(parts))))
        hint = "↑↓ scroll · ? / esc back"
        if Text(hint).cell_len > layout["inner"]:
            hint = "? / esc back" if layout["inner"] >= 12 else "esc"
        parts.append(fitted(Text(hint, style="dim"), layout["inner"]))
        return self._panel(parts, layout, "tree · shortcuts")

    # ------------------------------------------------------------------ #
    # Input and actions; browsing does not check out or edit stored history
    # ------------------------------------------------------------------ #
    def on_interrupt(self) -> None:
        self.stop()

    def on_resize(self, size: tuple[int, int]) -> None:
        if not self.full_preview and not self._layout()["split"]:
            self.pane_focus = "tree"

    def _toggle_fold(self, row: TreeRow) -> None:
        if self.search_query:
            self.flash = ("Clear search to fold branches.", "dim")
            return
        if row.run:
            self.view.folded_runs.discard(row.index)
            self._select_node(row.index)
        elif row.index in self.view.run_start:
            start = self.view.run_start[row.index]
            self.view.folded_runs.add(start)
            self.selected, self.selected_run = start, True
        elif self.nodes[row.index].children:
            if self.nodes[row.index].on_active_path:
                self.flash = ("The current path stays visible; fold an earlier run or an inactive branch.", "dim")
            elif row.index in self.view.folded_branches:
                self.view.folded_branches.remove(row.index)
            else:
                self.view.folded_branches.add(row.index)
        self._rows_cache = None

    def on_key(self, key: str, repeat: int) -> None:
        if self.search_active:
            self._on_key_search(key, repeat)
            return
        if isinstance(key, TextInput):
            self.arm_delete_id, self.flash = None, None
            return
        layout = self._layout()
        if self.help_open:
            if key in ("?", "ESC", "q", "Q"):
                self.help_open = False
            else:
                self.help_offset = apply_scroll_keys(key, repeat, offset=self.help_offset,
                    total=len(self._help_lines(layout["inner"])),
                    body_rows=max(1, layout["height"] - menu_frame_rows() - 1))
            return
        if key == "ESC" and self.arm_delete_id is not None:
            self.arm_delete_id, self.flash = None, None
            return
        if key not in ("d", "D"):
            self.arm_delete_id, self.flash = None, None
        if key == "?":
            self.help_open, self.help_offset = True, 0
            return
        if key == "CTRL_F":
            self.full_preview = False
            self.pane_focus = "tree"
            self.search_active = True
            initialize_text_editor(self.search_editor, self.search_query)
            return

        row = self._selected_row()
        if self.full_preview:
            if key in ("p", "P", "ESC", "LEFT", "q", "Q", "TAB"):
                self._close_preview()
                if key == "TAB":
                    self.pane_focus = "tree"
            else:
                self._scroll_preview(key, repeat, layout)
            return
        if key in ("p", "P") and row is not None:
            self._open_preview()
            return
        if key == "TAB" and row is not None:
            if layout["split"]:
                self.pane_focus = "preview" if self.pane_focus == "tree" else "tree"
            else:
                self._open_preview()
            return
        if self.pane_focus == "preview":
            if key in ("ESC", "LEFT", "q", "Q"):
                self.pane_focus = "tree"
            else:
                self._scroll_preview(key, repeat, layout)
            return
        if key in ("ESC", "q", "Q"):
            if self.search_query:
                self.search_query = ""
                self._update_search()
            else:
                self.stop()
            return
        if row is None:
            return

        rows = self._rows()
        position = rows.index(row)
        nav_key = {"MOUSE_WHEEL_UP": "UP", "MOUSE_WHEEL_DOWN": "DOWN",
                   "MOUSE_WHEEL_PAGEUP": "PAGEUP", "MOUSE_WHEEL_PAGEDOWN": "PAGEDOWN"}.get(key, key)
        nav = apply_selection_keys(nav_key, repeat, selected=position, total=len(rows),
                                   page=max(1, layout["tree_height"] - layout["heading"] - 1))
        node = self.nodes[row.index]
        if nav is not None:
            self._select_row(rows[nav])
        elif key == "LEFT":
            if node.parent is not None:
                self._select_node(self.index_by_id[id(node.parent)])
        elif key == "RIGHT":
            if row.run:
                self._toggle_fold(row)
            elif node.children:
                self._select_node(self.index_by_id[id(node.children[0])])
        elif key in ("ENTER", "o", "O"):
            if row.run:
                self._toggle_fold(row)
            elif node.children:
                self._select_node(self.index_by_id[id(leaf_of(node))])
            else:
                self.outcome = TreeOutcome("open", node.frame_id, None)
                self.stop()
        elif key == " ":
            self._toggle_fold(row)
        elif not row.run:
            if key in ("f", "F"):
                self.outcome = TreeOutcome("fork", node.frame_id, None)
                self.stop()
            elif key in ("e", "E") and not node.children:
                self.outcome = TreeOutcome("edit", parent_id_of(node), node.original_prompt)
                self.stop()
            elif key in ("d", "D"):
                self._on_delete(node)

    def _on_delete(self, node) -> None:
        if node.on_active_path:
            self.arm_delete_id = None
            self.flash = ("Can't delete the active branch — it's the live thread.", "bold yellow")
            return
        if self.arm_delete_id != node.frame_id:
            self.arm_delete_id = node.frame_id
            count = self.view.sizes[self.index_by_id[id(node)]]
            self.flash = (f"Delete {count} prompt{'s' if count != 1 else ''}? Press d again to confirm · any other key cancels.", "bold red")
            return
        self.arm_delete_id = None
        try:
            if self.history_file is None or not delete_subtree(self.history_file, node_id=node.frame_id):
                self.flash = ("Couldn't delete this branch; reopen the tree to refresh it.", "bold red")
                return
            target = node.parent.frame_id if node.parent is not None else None
            self._reload(select_id=target)
        except (StorageError, OSError) as exc:
            self.flash = (f"Couldn't delete branch: {exc}", "bold red")
            return
        self.flash = ("Branch deleted.", "green")

    def _reload(self, *, select_id: str | None = None) -> None:
        folded = {self.nodes[index].frame_id for index in self.view.folded_branches}
        expanded = {self.nodes[index].frame_id for index in self.view.runs if index not in self.view.folded_runs}
        self.model = load_session_tree(self.history_file)
        self._reset_view()
        self.view.folded_branches = {i for i, node in enumerate(self.nodes) if node.frame_id in folded}
        self.view.folded_runs.difference_update(i for i in self.view.runs if self.nodes[i].frame_id in expanded)
        self.selected = next((i for i, node in enumerate(self.nodes) if node.frame_id == select_id),
                             next((i for i, node in enumerate(self.nodes) if node.is_active_leaf), 0))
        self.selected_run = False
        self.search_text = None
        self.preview_cache.clear()
        self.preview_positions.clear()
        self._update_search()
        self.offset = 0


def run_tree_screen(session_context, config=None) -> TreeOutcome:
    """Build the tree for the active session and run the interactive view."""
    model = load_session_tree(session_context.history_file)
    screen = TreeBrowserScreen(model=model, history_file=session_context.history_file)
    screen.run()
    return screen.outcome
