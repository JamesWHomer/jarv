"""Pure logic for the session prompt-tree: frames, branches, and checkout.

A jarv session is stored as a flat, linear history (one root->leaf path). To let
the user branch the conversation, every *other* path is kept in a sidecar
(``branches-<hash>.json``) as a flat list of off-spine *frames*, each tagged with
the id of its parent frame. This module turns those two stores into a navigable
tree and performs the single mutation the ``/tree`` view needs -- :func:`checkout`
-- which re-picks which root->leaf path is "active" while preserving every frame.

It never touches the agent's hot loop: the agent reloads the active history from
disk each turn (see ``run_agent``), so a checkout simply changes what the next
turn reads. The design generalizes the existing ``/undo`` + ``/redo`` machinery,
which already splits history into frames and stashes them in a sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .history import (
    branches_file_for,
    load_branches,
    load_history,
    new_frame_id,
    redo_file_for,
    save_branches,
    save_history,
)
from .session_render import _history_content_to_str

#: Sentinel parent id meaning "attaches at the root" (an alternative first prompt).
ROOT = ""


# --------------------------------------------------------------------------- #
# Frame splitting
# --------------------------------------------------------------------------- #
def _is_user(item) -> bool:
    return isinstance(item, dict) and str(item.get("role", "")).lower() == "user"


def iter_frames(items: list) -> list[list]:
    """Split a flat history into frames (one prompt + its full response).

    A frame runs from one user message up to (but not including) the next one.
    Every item is covered with no loss, so the frames concatenate back to the
    input. For well-formed histories (which always start with a user message)
    this matches ``context_budget.iter_turn_ranges``; any rare leading non-user
    items simply ride with the first frame.
    """
    frames: list[list] = []
    n = len(items)
    i = 0
    while i < n:
        start = i
        i += 1
        while i < n and not _is_user(items[i]):
            i += 1
        frames.append(items[start:i])
    return frames


def _frame_prompt(frame: list) -> str:
    for item in frame:
        if _is_user(item):
            # Collapse whitespace so a multi-line prompt stays a single row in
            # the tree view (mirrors _frame_response, which flattens too).
            return " ".join(_history_content_to_str(item.get("content", "")).split())
    return ""


def _frame_response(frame: list) -> str:
    for item in frame:
        if isinstance(item, dict) and str(item.get("role", "")).lower() == "assistant":
            text = _history_content_to_str(item.get("content", "")).strip()
            if text:
                return text.replace("\n", " ")
    tools = [
        item.get("name", "tool")
        for item in frame
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    if tools:
        return "↳ " + ", ".join(dict.fromkeys(tools))
    return ""


# --------------------------------------------------------------------------- #
# Tree model
# --------------------------------------------------------------------------- #
@dataclass
class TreeNode:
    frame_id: str
    items: list
    prompt_text: str
    response_preview: str
    depth: int
    parent: "TreeNode | None" = None
    children: list["TreeNode"] = field(default_factory=list)
    on_active_path: bool = False
    is_active_leaf: bool = False


@dataclass
class TreeModel:
    roots: list[TreeNode]          # top-level nodes (active root first)
    nodes: list[TreeNode]          # flattened depth-first (display order)
    by_id: dict[str, TreeNode]
    active_path: list[TreeNode]    # root->active-leaf

    def find(self, frame_id: str) -> TreeNode | None:
        return self.by_id.get(frame_id)


def _frame_id(frame: list, fallback: str) -> str:
    for item in frame:
        if _is_user(item):
            fid = item.get("id")
            if isinstance(fid, str) and fid:
                return fid
            break
    return fallback


def build_tree(history: list, branches: list[dict]) -> TreeModel:
    """Assemble the prompt tree from the active history and stored branches.

    Frame ids come from each prompt's ``id`` field; legacy frames with no id get
    a deterministic positional fallback (``a<i>`` for the active path, ``b<i>``
    for stored frames) so the same id is produced every time the same on-disk
    state is built -- which is what lets the view hand a ``frame_id`` back to
    :func:`checkout` unchanged.
    """
    by_id: dict[str, TreeNode] = {}
    nodes_in_order: list[TreeNode] = []

    # Active path: a simple chain rooted at the first frame.
    active_frames = iter_frames(history)
    active_nodes: list[TreeNode] = []
    prev: TreeNode | None = None
    for i, frame in enumerate(active_frames):
        node = TreeNode(
            frame_id=_frame_id(frame, f"a{i}"),
            items=frame,
            prompt_text=_frame_prompt(frame),
            response_preview=_frame_response(frame),
            depth=0,
            parent=prev,
            on_active_path=True,
        )
        if prev is not None:
            prev.children.append(node)
        active_nodes.append(node)
        by_id[node.frame_id] = node
        prev = node
    if active_nodes:
        active_nodes[-1].is_active_leaf = True

    # Off-spine frames: attach by parent id. Iterate to a fixpoint so a branch
    # whose parent is itself a (later-listed) stored frame still attaches.
    pending: list[tuple[str, TreeNode]] = []
    for i, record in enumerate(branches):
        items = record.get("items") or []
        node = TreeNode(
            frame_id=_frame_id(items, f"b{i}"),
            items=items,
            prompt_text=_frame_prompt(items),
            response_preview=_frame_response(items),
            depth=0,
            on_active_path=False,
        )
        by_id.setdefault(node.frame_id, node)
        parent_id = record.get("parent_frame_id", ROOT)
        pending.append((parent_id if isinstance(parent_id, str) else ROOT, node))

    roots: list[TreeNode] = list(active_nodes[:1])
    progressed = True
    while pending and progressed:
        progressed = False
        still: list[tuple[str, TreeNode]] = []
        for parent_id, node in pending:
            if parent_id == ROOT:
                roots.append(node)
                progressed = True
            elif parent_id in by_id:
                parent = by_id[parent_id]
                node.parent = parent
                parent.children.append(node)
                progressed = True
            else:
                still.append((parent_id, node))
        pending = still
    # Anything still dangling (corrupt parent ref) becomes a root so it is never
    # lost from the tree or from a later checkout's frame set.
    for _parent_id, node in pending:
        roots.append(node)

    # Order each node's children so the active continuation leads, then flatten
    # depth-first into display order with correct depth.
    def _order(node: TreeNode) -> None:
        node.children.sort(key=lambda c: (not c.on_active_path,))

    def _walk(node: TreeNode, depth: int) -> None:
        node.depth = depth
        nodes_in_order.append(node)
        _order(node)
        for child in node.children:
            _walk(child, depth + 1)

    roots.sort(key=lambda c: (not c.on_active_path,))
    for root in roots:
        _walk(root, 0)

    active_path = [n for n in active_nodes]
    return TreeModel(roots=roots, nodes=nodes_in_order, by_id=by_id, active_path=active_path)


# --------------------------------------------------------------------------- #
# Navigation helpers
# --------------------------------------------------------------------------- #
def leaf_of(node: TreeNode) -> TreeNode:
    """Descend to the tip of ``node``'s line (active continuation preferred)."""
    cur = node
    while cur.children:
        cur = cur.children[0]
    return cur


def parent_id_of(node: TreeNode) -> str:
    return node.parent.frame_id if node.parent is not None else ROOT


def ancestors_and_self(node: TreeNode) -> list[TreeNode]:
    chain: list[TreeNode] = []
    cur: TreeNode | None = node
    while cur is not None:
        chain.append(cur)
        cur = cur.parent
    chain.reverse()
    return chain


# --------------------------------------------------------------------------- #
# The single mutation
# --------------------------------------------------------------------------- #
def _set_frame_id(frame: list) -> str:
    """Ensure the frame's leading user item has a real id; return it."""
    for item in frame:
        if _is_user(item):
            fid = item.get("id")
            if not (isinstance(fid, str) and fid):
                fid = new_frame_id()
                item["id"] = fid
            return fid
    # No user item (rare leading-preamble frame): synthesize a carrier id.
    return new_frame_id()


def checkout(history_file: Path, *, leaf_id: str) -> bool:
    """Make ``leaf_id`` the active leaf; stash every other frame as a branch.

    Rebuilds the active history as ``root -> leaf_id`` and writes all remaining
    frames to the branches sidecar (each tagged with its parent frame's id). The
    full set of frames is invariant -- no prompt or response is ever dropped.
    ``leaf_id == ROOT`` empties the active history (used when editing a root
    prompt). Returns ``True`` if anything changed on disk.
    """
    history = load_history(history_file)
    branches_path = branches_file_for(history_file)
    model = build_tree(history, load_branches(branches_path))

    if leaf_id == ROOT:
        target: TreeNode | None = None
        spine: list[TreeNode] = []
    else:
        target = model.find(leaf_id)
        if target is None:
            return False
        # No-op: already sitting on this exact leaf.
        if target.is_active_leaf:
            return False
        spine = ancestors_and_self(target)

    spine_set = {id(n) for n in spine}
    off_spine = [n for n in model.nodes if id(n) not in spine_set]

    # Assign real ids before computing parent pointers so every reference is stable.
    for node in spine:
        node.frame_id = _set_frame_id(node.items)
    for node in off_spine:
        node.frame_id = _set_frame_id(node.items)

    new_history: list = []
    for node in spine:
        new_history.extend(node.items)

    new_frames = [
        {
            "parent_frame_id": node.parent.frame_id if node.parent is not None else ROOT,
            "items": node.items,
        }
        for node in off_spine
    ]

    save_history(new_history, history_file)
    save_branches(new_frames, branches_path)
    # The linear redo stack no longer describes this path; drop it.
    redo_path = redo_file_for(history_file)
    if redo_path.exists():
        try:
            redo_path.unlink()
        except OSError:
            pass
    return True


def delete_subtree(history_file: Path, *, node_id: str) -> bool:
    """Permanently drop an off-spine prompt and everything beneath it.

    Only branches (nodes off the active path) can be deleted -- the active path is
    the live conversation. The active history is untouched apart from backfilling
    ids so surviving branches keep valid parent references. Returns ``True`` if a
    subtree was removed.
    """
    history = load_history(history_file)
    branches_path = branches_file_for(history_file)
    model = build_tree(history, load_branches(branches_path))

    target = model.find(node_id)
    if target is None or target.on_active_path:
        return False

    doomed: set[int] = set()
    stack = [target]
    while stack:
        node = stack.pop()
        doomed.add(id(node))
        stack.extend(node.children)

    survivors = [n for n in model.nodes if not n.on_active_path and id(n) not in doomed]

    # A survivor's parent is always on the active path or another survivor (we drop
    # whole subtrees), so backfilling ids on both keeps every reference resolvable.
    for node in model.active_path:
        node.frame_id = _set_frame_id(node.items)
    for node in survivors:
        node.frame_id = _set_frame_id(node.items)

    new_history: list = []
    for node in model.active_path:
        new_history.extend(node.items)
    new_frames = [
        {
            "parent_frame_id": node.parent.frame_id if node.parent is not None else ROOT,
            "items": node.items,
        }
        for node in survivors
    ]

    save_history(new_history, history_file)
    save_branches(new_frames, branches_path)
    return True
