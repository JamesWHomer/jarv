import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jarv.context_budget import iter_turn_ranges
from jarv.history import branches_file_for, load_branches, load_history, save_branches, save_history
from jarv.session_tree import (
    ROOT,
    _frame_prompt,
    build_tree,
    checkout,
    delete_subtree,
    iter_frames,
    leaf_of,
    parent_id_of,
)


def frame(prompt, resp="ok", fid=None):
    user = {"role": "user", "content": prompt}
    if fid is not None:
        user["id"] = fid
    return [user, {"role": "assistant", "content": resp}]


def history_of(*frames):
    items: list = []
    for f in frames:
        items.extend(f)
    return items


def all_prompts(history_file):
    """Every prompt across the active path and the branch sidecar (sorted)."""
    history = load_history(history_file)
    branches = load_branches(branches_file_for(history_file))
    prompts = [_frame_prompt(f) for f in iter_frames(history)]
    prompts += [_frame_prompt(rec["items"]) for rec in branches]
    return sorted(prompts)


def active_prompts(history_file):
    return [_frame_prompt(f) for f in iter_frames(load_history(history_file))]


class FrameSplittingTests(unittest.TestCase):
    def test_matches_iter_turn_ranges_for_wellformed_history(self):
        history = history_of(frame("a"), frame("b"), frame("c"))
        ranges = iter_turn_ranges(history)
        frames = iter_frames(history)
        rebuilt = [history[s:e] for s, e in ranges]
        self.assertEqual(frames, rebuilt)

    def test_is_lossless(self):
        history = history_of(frame("a"), frame("b"))
        flat: list = []
        for f in iter_frames(history):
            flat.extend(f)
        self.assertEqual(flat, history)

    def test_empty_history(self):
        self.assertEqual(iter_frames([]), [])
        model = build_tree([], [])
        self.assertEqual(model.nodes, [])

    def test_prompt_collapses_to_single_line(self):
        # A multi-line prompt must flatten so it stays one row in the tree view.
        f = frame("line one\nline two\n  indented", resp="ok")
        self.assertEqual(_frame_prompt(f), "line one line two indented")


class BuildTreeTests(unittest.TestCase):
    def test_linear_history_is_a_spine(self):
        history = history_of(frame("a", fid="f0"), frame("b", fid="f1"), frame("c", fid="f2"))
        model = build_tree(history, [])
        self.assertEqual([n.prompt_text for n in model.nodes], ["a", "b", "c"])
        self.assertTrue(all(n.on_active_path for n in model.nodes))
        self.assertTrue(model.nodes[-1].is_active_leaf)
        self.assertEqual(len(model.active_path), 3)

    def test_branch_attaches_under_parent(self):
        history = history_of(frame("a", fid="f0"), frame("b", fid="f1"))
        branches = [{"parent_frame_id": "f0", "items": frame("alt", fid="x0")}]
        model = build_tree(history, branches)
        f0 = model.find("f0")
        self.assertEqual(len(f0.children), 2)
        labels = {c.prompt_text for c in f0.children}
        self.assertEqual(labels, {"b", "alt"})
        # Active continuation sorts first.
        self.assertTrue(f0.children[0].on_active_path)


class CheckoutTests(unittest.TestCase):
    def _hist(self, tmp):
        hf = Path(tmp) / "history-test.json"
        save_history(history_of(frame("a", fid="f0"), frame("b", fid="f1"), frame("c", fid="f2")), hf)
        return hf

    def test_fork_truncates_and_stashes_tail(self):
        with TemporaryDirectory() as tmp:
            hf = self._hist(tmp)
            before = all_prompts(hf)
            self.assertTrue(checkout(hf, leaf_id="f1"))
            self.assertEqual(active_prompts(hf), ["a", "b"])
            # No prompt was lost; "c" now lives in the sidecar.
            self.assertEqual(all_prompts(hf), before)
            branches = load_branches(branches_file_for(hf))
            self.assertEqual(len(branches), 1)
            self.assertEqual(branches[0]["parent_frame_id"], "f1")

    def test_open_roundtrip_preserves_every_prompt(self):
        with TemporaryDirectory() as tmp:
            hf = self._hist(tmp)
            # Fork at f1, then simulate the user sending a new prompt on the fork.
            checkout(hf, leaf_id="f1")
            new = load_history(hf) + frame("b2", fid="f1b")
            save_history(new, hf)
            self.assertEqual(sorted(all_prompts(hf)), ["a", "b", "b2", "c"])

            # Resume the original branch (whose tip is f2).
            model = build_tree(load_history(hf), load_branches(branches_file_for(hf)))
            f2 = model.find("f2")
            self.assertTrue(checkout(hf, leaf_id=leaf_of(f2).frame_id))
            self.assertEqual(active_prompts(hf), ["a", "b", "c"])
            self.assertEqual(sorted(all_prompts(hf)), ["a", "b", "b2", "c"])

    def test_open_active_leaf_is_noop(self):
        with TemporaryDirectory() as tmp:
            hf = self._hist(tmp)
            self.assertFalse(checkout(hf, leaf_id="f2"))
            self.assertEqual(active_prompts(hf), ["a", "b", "c"])

    def test_edit_uses_parent_and_preserves_subtree(self):
        with TemporaryDirectory() as tmp:
            hf = self._hist(tmp)
            model = build_tree(load_history(hf), [])
            node = model.find("f1")
            self.assertEqual(parent_id_of(node), "f0")
            self.assertTrue(checkout(hf, leaf_id=parent_id_of(node)))
            self.assertEqual(active_prompts(hf), ["a"])
            self.assertEqual(sorted(all_prompts(hf)), ["a", "b", "c"])

    def test_checkout_root_empties_active_history(self):
        with TemporaryDirectory() as tmp:
            hf = self._hist(tmp)
            self.assertTrue(checkout(hf, leaf_id=ROOT))
            self.assertEqual(load_history(hf), [])
            self.assertEqual(sorted(all_prompts(hf)), ["a", "b", "c"])

    def test_legacy_frames_get_ids_on_checkout(self):
        with TemporaryDirectory() as tmp:
            hf = Path(tmp) / "history-legacy.json"
            save_history(history_of(frame("a"), frame("b"), frame("c")), hf)  # no ids
            # The view builds deterministic positional ids; "a1" is the 2nd frame.
            self.assertTrue(checkout(hf, leaf_id="a1"))
            active = load_history(hf)
            user_ids = [i["id"] for i in active if i.get("role") == "user"]
            self.assertTrue(all(isinstance(uid, str) and uid for uid in user_ids))
            branches = load_branches(branches_file_for(hf))
            # The stashed tail points at a real (now-persisted) parent id.
            self.assertIn(branches[0]["parent_frame_id"], user_ids)

    def test_unknown_leaf_is_noop(self):
        with TemporaryDirectory() as tmp:
            hf = self._hist(tmp)
            self.assertFalse(checkout(hf, leaf_id="does-not-exist"))


class DeleteSubtreeTests(unittest.TestCase):
    def _session(self, tmp):
        """Active [a, b] with branch 'c' under b, and 'd' nested under 'c'."""
        hf = Path(tmp) / "history-del.json"
        save_history(history_of(frame("a", fid="f0"), frame("b", fid="f1")), hf)
        save_branches(
            [
                {"parent_frame_id": "f1", "items": frame("c", fid="c0")},
                {"parent_frame_id": "c0", "items": frame("d", fid="d0")},
            ],
            branches_file_for(hf),
        )
        return hf

    def test_delete_leaf_branch(self):
        with TemporaryDirectory() as tmp:
            hf = self._session(tmp)
            self.assertTrue(delete_subtree(hf, node_id="d0"))
            self.assertEqual(active_prompts(hf), ["a", "b"])
            self.assertEqual(all_prompts(hf), ["a", "b", "c"])  # only d removed

    def test_delete_subtree_removes_descendants(self):
        with TemporaryDirectory() as tmp:
            hf = self._session(tmp)
            self.assertTrue(delete_subtree(hf, node_id="c0"))
            self.assertEqual(active_prompts(hf), ["a", "b"])
            self.assertEqual(all_prompts(hf), ["a", "b"])  # c and nested d both gone

    def test_cannot_delete_active_path(self):
        with TemporaryDirectory() as tmp:
            hf = self._session(tmp)
            self.assertFalse(delete_subtree(hf, node_id="f0"))
            self.assertEqual(all_prompts(hf), ["a", "b", "c", "d"])

    def test_delete_unknown_is_noop(self):
        with TemporaryDirectory() as tmp:
            hf = self._session(tmp)
            self.assertFalse(delete_subtree(hf, node_id="nope"))
            self.assertEqual(all_prompts(hf), ["a", "b", "c", "d"])

    def test_sibling_branch_survives(self):
        with TemporaryDirectory() as tmp:
            hf = Path(tmp) / "history-sib.json"
            save_history(history_of(frame("a", fid="f0")), hf)
            save_branches(
                [
                    {"parent_frame_id": "f0", "items": frame("x", fid="x0")},
                    {"parent_frame_id": "f0", "items": frame("y", fid="y0")},
                ],
                branches_file_for(hf),
            )
            self.assertTrue(delete_subtree(hf, node_id="x0"))
            self.assertEqual(all_prompts(hf), ["a", "y"])


class BranchSidecarTests(unittest.TestCase):
    def test_roundtrip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "branches-x.json"
            records = [{"parent_frame_id": "p", "items": frame("z", fid="z0")}]
            save_branches(records, path)
            self.assertEqual(load_branches(path), records)

    def test_corrupt_file_returns_empty(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "branches-x.json"
            path.write_text("{ not json", encoding="utf-8")
            self.assertEqual(load_branches(path), [])

    def test_missing_file_returns_empty(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(load_branches(Path(tmp) / "nope.json"), [])


if __name__ == "__main__":
    unittest.main()
