from unittest.mock import patch

from jarv.agent_ui import _dispatch_spawn_with_ui
from jarv.artifacts import ArtifactStore
from jarv.config import DEFAULT_CONFIG
from jarv.orchestrator import AgentNode


class _FakeLive:
    def __init__(self, renderable, **_kwargs):
        self.renderable = renderable

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def update(self, renderable):
        self.renderable = renderable


class _FakeUI:
    def __init__(self):
        self.finished_states = []

    def show_tool_card(self, renderable):
        self.finished_states.append(renderable.finished)


def test_print_mode_spawn_returns_model_visible_output():
    root = AgentNode(
        label="root",
        depth=0,
        parent_label=None,
        task="root",
        sterile=False,
    )
    expected = '[{"label": "child", "status": "done", "tldr": "ok"}]'

    with (
        patch("jarv.agent_ui.Live", _FakeLive),
        patch("jarv.agent_ui.spawn_tool_output", return_value=expected),
        patch("jarv.agent_ui.print_mode_spacer") as spacer,
    ):
        output = _dispatch_spawn_with_ui(
            {"children": [{"label": "child", "task": "work"}]},
            root,
            ArtifactStore(),
            client=None,
            config=DEFAULT_CONFIG,
        )

    assert output == expected
    spacer.assert_called_once_with(DEFAULT_CONFIG)


def test_headsup_spawn_panel_reports_finished_after_children_complete():
    root = AgentNode(
        label="root",
        depth=0,
        parent_label=None,
        task="root",
        sterile=False,
    )
    ui = _FakeUI()

    def finish_child(*_args, spawn_observer=None, **_kwargs):
        spawn_observer.on_child_done(
            "root",
            "child",
            {"label": "child", "status": "done", "tldr": "ok"},
        )
        return '[{"label": "child", "status": "done", "tldr": "ok"}]'

    with patch("jarv.agent_ui.spawn_tool_output", side_effect=finish_child):
        _dispatch_spawn_with_ui(
            {"children": [{"label": "child", "task": "work"}]},
            root,
            ArtifactStore(),
            client=None,
            config=DEFAULT_CONFIG,
            ui=ui,
        )

    assert ui.finished_states[0] is False
    assert ui.finished_states[-1] is True
