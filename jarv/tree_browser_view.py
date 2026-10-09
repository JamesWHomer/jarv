"""A foldable projection of a session tree; stored nodes are never modified."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TreeRow:
    index: int
    connector: str
    run: tuple[int, ...] = ()
    hidden: int = 0


class TreeView:
    def __init__(self, model):
        self.nodes = model.nodes
        self.index_by_id = {id(node): index for index, node in enumerate(self.nodes)}
        self.roots = [self.index_by_id[id(node)] for node in model.roots]
        self.children = [
            [self.index_by_id[id(child)] for child in node.children]
            for node in self.nodes
        ]
        # The model is in preorder, so each subtree occupies one contiguous span.
        self.sizes = [1] * len(self.nodes)
        for index in reversed(range(len(self.nodes))):
            self.sizes[index] += sum(self.sizes[child] for child in self.children[index])

        protected = set()
        for index, node in enumerate(self.nodes):
            if node.is_active_leaf:
                protected.add(index)
                if node.parent is not None:
                    protected.add(self.index_by_id[id(node.parent)])

        self.runs: dict[int, tuple[int, ...]] = {}
        self.run_start: dict[int, int] = {}
        visited = set()
        for index in range(len(self.nodes)):
            run = []
            current = index
            while current not in visited and current not in protected and len(self.children[current]) == 1:
                visited.add(current)
                run.append(current)
                current = self.children[current][0]
            if len(run) >= 3:
                self.runs[index] = tuple(run)
                self.run_start.update((member, index) for member in run)

        # Compact long shared history on opening; other branches remain expanded.
        self.folded_runs = {index for index in self.runs if self.nodes[index].on_active_path}
        self.folded_branches: set[int] = set()

    def reveal(self, index: int) -> None:
        """Expose an actual prompt and its parent after a structural jump."""
        targets = [index]
        parent = self.nodes[index].parent
        if parent is not None:
            targets.append(self.index_by_id[id(parent)])
        for target in targets:
            self.folded_runs.discard(self.run_start.get(target))
            self.folded_branches.difference_update(
                branch for branch in tuple(self.folded_branches)
                if branch < target < branch + self.sizes[branch]
            )

    def with_ancestors(self, matches: set[int]) -> set[int]:
        """Retain the exact path to each search result in linear time."""
        included = set(matches)
        for index in reversed(range(len(self.nodes))):
            if index in included and self.nodes[index].parent is not None:
                included.add(self.index_by_id[id(self.nodes[index].parent)])
        return included

    def rows(self, included: set[int] | None = None) -> list[TreeRow]:
        rows = []
        stack = [(index, (), False) for index in reversed(self.roots)]
        while stack:
            index, trail, shortened = stack.pop()
            if included is not None and index not in included:
                continue
            connector = ""
            if trail:
                connector = ("… " if shortened else "") + "".join(
                    "   " if last else "│  " for last in trail[:-1]
                ) + ("└─ " if trail[-1] else "├─ ")

            run = self.runs[index] if included is None and index in self.folded_runs else ()
            hidden = self.sizes[index] - 1 if included is None and index in self.folded_branches else 0
            rows.append(TreeRow(index, connector, run, hidden))
            children = [] if hidden else self.children[run[-1] if run else index]
            if included is not None:
                children = [child for child in children if child in included]
            stack.extend(
                (child, (trail + (position == len(children) - 1,))[-24:], shortened or len(trail) >= 24)
                for position, child in reversed(list(enumerate(children)))
            )
        return rows
