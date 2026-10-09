"""Interactive command continuations contribute to the single final summary."""

from jarv import agent
from jarv.config import DEFAULT_CONFIG
from jarv.headsup import HeadsupAgentUI
from jarv.provider import StreamDone, TextDelta, ToolCallDone
from jarv.shell import InteractiveCommandSnapshot


def test_interactive_continuations_are_silent_until_user_turn_completes(
    monkeypatch, headsup_app_factory,
):
    app = headsup_app_factory()

    def summaries():
        return [entry.renderable.plain for entry in app.entries if entry.kind == "usage"]

    class FakeProcess:
        def __init__(self):
            self.inputs = []
            self.snapshots = iter([
                InteractiveCommandSnapshot("menu", "Choose:\n", "", None, exited=False),
                InteractiveCommandSnapshot("menu", "Name:\n", "", None, exited=False),
                InteractiveCommandSnapshot("menu", "choice=3 name=Ada\n", "", 0, exited=True),
            ])

        def write_stdin(self, text):
            assert summaries() == []
            self.inputs.append(text)

        def wait_until_idle(self, **_kwargs):
            assert summaries() == []
            return next(self.snapshots)

        def kill_tree(self):
            pass

    process = FakeProcess()
    rounds = []

    def stream(*_args, **_kwargs):
        assert summaries() == []
        rounds.append(True)
        round_number = len(rounds)
        if round_number == 1:
            yield ToolCallDone("fc_1", "call_1", "run_command", '{"command":"menu"}')
        else:
            yield TextDelta({2: "3", 3: "Ada", 4: "Done."}[round_number])
        yield StreamDone({"usage": {
            "input_tokens": round_number * 100,
            "output_tokens": round_number * 10,
        }})

    monkeypatch.setattr(agent, "stream_response", stream)
    monkeypatch.setattr(agent, "check_command", lambda *a, **kw: (True, ""))
    monkeypatch.setattr(agent.InteractiveCommandProcess, "start", lambda *a, **kw: process)
    monkeypatch.setattr(
        agent, "_prepare_client_and_instructions", lambda *a, **kw: (object(), "system")
    )

    result = agent.run_agent(
        "Run the menu",
        {**DEFAULT_CONFIG, "interactive_commands": True, "turn_summary": True},
        client=object(), incognito=True, heads_up=True, ui=HeadsupAgentUI(app),
    )

    assert result.error is None
    assert result.turns == 4
    assert result.text == "Done."
    assert process.inputs == ["3\n", "Ada\n"]
    assert len(summaries()) == 1
    assert "1,000 in" in summaries()[0]
    assert "100 out" in summaries()[0]
    assert "1,100 total" in summaries()[0]
    assert app.entries[-2].kind == "assistant"
    assert app.entries[-1].kind == "usage"
