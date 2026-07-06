import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from rich.control import Control, ControlType
from rich.live import Live
from rich.live_render import LiveRender
from rich.markdown import Markdown
from rich.markup import escape
from rich.segment import Segment
from rich.text import Text

from .config import DEFAULT_CONFIG, get_setting
from .context_budget import build_input, trim_turn_input
from .cancellation import CancellationToken, TurnCancelled, cancel_token_on_sigint
from .display import (
    console,
    flatten_headings,
    track_live_display,
)
from .history import (
    artifact_file_for,
    forget_current_session,
    get_shell_name,
    history_metadata,
    load_history,
    new_frame_id,
    prepare_session_context,
    reads_file_for,
    redo_file_for,
    save_history,
)
from .artifacts import ArtifactStore, load_artifact_store, save_artifact_store
from .provider import (
    create_client,
    ProviderError,
    RetryableStreamError,
    ReasoningDone,
    ReasoningStarted,
    TextDelta,
    ToolCallDone,
    ToolCallStarted,
    provider_response_notice,
    stream_response,
)
from .response_items import to_response_input_item
from .response_items import status_history_item
from .orchestrator import (
    ASK_USER_TOOL,
    PendingRunCommand,
    RUN_COMMAND_TOOL,
    RunCommandDispatchResult,
    SPAWN_TOOL,
    AgentNode,
    SpawnObserver,
    ToolExecutionHooks,
    execute_tool_calls,
    filter_enabled_tools,
    history_has_web_search_read_nudge,
    parse_spawn_children,
    prepare_run_command,
    spawn_tool_output,
)
from .edit_tool import EDIT_TOOL, dispatch_edit_tool
from .read_tool import read_tool_for_config
from .retained_outputs import (
    RetainedOutputStore,
    load_retained_output_store,
    save_retained_output_store,
)
from .tool_outputs import ToolOutput
from .turn_loop import StreamCollection, collect_stream_response, run_tool_execution_round
from .turn_records import (
    append_reasoning_input_items,
    append_tool_result_input_items,
    stream_usage_output_text,
)
from .usage import (
    estimate_context_breakdown,
    format_cost,
    format_int,
    load_usage,
    record_response_usage,
    usage_cost_summary,
    usage_file_for,
)
from .web import WEB_SEARCH_TOOL


def build_agent_tools(config: dict) -> list[dict]:
    tools = [
        RUN_COMMAND_TOOL,
        WEB_SEARCH_TOOL,
        SPAWN_TOOL,
        read_tool_for_config(config),
        EDIT_TOOL,
        ASK_USER_TOOL,
    ]
    return filter_enabled_tools(tools, config)


TOOLS = build_agent_tools(DEFAULT_CONFIG)


def resolve_tool_call_display(config: dict, *, heads_up: bool) -> str:
    mode = get_setting(config, "tool_call_display")
    if mode == "auto":
        return "fullscreen" if heads_up else "print"
    return str(mode)


from .agent_ui import (
    InPlaceLive,
    InteractiveCommandCard,
    ResponseWaitIndicator,
    StreamingMarkdownPreview,
    TailMarkdown,
    ToolActivityIndicator,
    _dispatch_ask_user,
    _dispatch_spawn_with_ui,
    _format_agent_usage_line,
    _print_agent_usage_if_enabled,
    _print_tool_card,
    _start_response_wait_indicator,
    _ui_call,
    get_system_info,
    print_mode_spacer,
    response_start_status,
    response_wait_label,
    tool_activity_complete_status,
    tool_activity_label,
    tool_complete_indicator,
    thought_complete_indicator,
)
from .context_budget import build_input
from .project_context import build_project_context
from .response_items import to_response_input_item
from .safety import check_command
from .shell import InteractiveCommandProcess


@dataclass(frozen=True)
class AgentRunResult:
    cancelled: bool = False
    prompt: str | None = None
    error: str | None = None


class SessionPersistence:
    """The single "write the turn to disk" path for a :func:`run_agent` run.

    ``run_agent`` loads the session stores once after session prep, then mutates
    ``history`` in place for the rest of the turn. This collaborator owns those
    stores so the normal turn end, the error flush, and the cancel checkpoint all
    persist through one method instead of three copies of the same writes. Fields
    are populated incrementally during prep (mirroring the run-local variables) so
    a failure part-way through prep persists exactly what was loaded so far -- the
    same behaviour as the closures it replaces. Incognito runs never touch disk.
    """

    def __init__(self, *, incognito: bool):
        self.incognito = incognito
        self.history: list = []
        self.session_context = None
        self.artifact_store = None
        self.artifact_file = None
        self.retained_store = None
        self.reads_file = None

    def save(self, *, clear_redo: bool = False) -> None:
        if self.incognito:
            return
        if self.session_context is not None:
            if clear_redo:
                redo_path = redo_file_for(self.session_context.history_file)
                if redo_path.exists():
                    redo_path.unlink()
            save_history(self.history, self.session_context.history_file)
        if self.artifact_store is not None and self.artifact_file is not None:
            save_artifact_store(self.artifact_store, self.artifact_file)
        if self.retained_store is not None and self.reads_file is not None:
            save_retained_output_store(self.retained_store, self.reads_file)

    def save_turn(self) -> None:
        """Persist the turn and drop any stale redo checkpoint."""
        self.save(clear_redo=True)


def _agent_check_run_command(prepared, config: dict, **kwargs):
    safety_level = config.get("command_safety", "risky")
    audit = config.get("audit", True)
    return check_command(
        prepared.cmd,
        safety_level,
        audit=audit,
        config=config,
        history=kwargs.get("safety_history"),
        usage_path=kwargs.get("usage_path"),
        session_id=kwargs.get("session_id"),
        cancellation_token=kwargs.get("cancellation_token"),
    )


from .interactive_command import (
    _attach_interactive_output_item,
    _continue_interactive_command,
    _finalize_interactive_record,
    _first_terminal_action,
    _format_elapsed_seconds,
    _format_finished_interactive_output,
    _interaction_marker_text,
    _interactive_check_in_seconds,
    _interactive_tool_call_reminder,
    _parse_terminal_control,
    _record_interactive_input,
    _run_command_final_prompt,
    _run_command_waiting_prompt,
    _terminal_action_display,
)


def _stop_interactive_card_live(live, live_depth_cm) -> None:
    """Paint a final frame, stop the inline ``Live``, and release its depth.

    No-op for the handles heads-up leaves as ``None``. Tolerant of double calls
    and of Rich raising during teardown so a failed stop never masks the real
    error on the cancel/exception paths.
    """
    if live is not None:
        try:
            live.refresh()
            live.stop()
        except Exception:
            pass
    if live_depth_cm is not None:
        try:
            live_depth_cm.__exit__(None, None, None)
        except Exception:
            pass


def _close_interactive_card_live(pending) -> None:
    """Close a pending command's held-open card ``Live`` exactly once."""
    if pending is None:
        return
    _stop_interactive_card_live(
        getattr(pending, "live", None),
        getattr(pending, "live_depth_cm", None),
    )
    pending.live = None
    pending.live_depth_cm = None


def _refresh_interactive_card(ui, pending) -> None:
    """Repaint the growing interactive card after the main thread mutates it.

    Heads-up re-renders the card into its live-tool slot; inline just nudges the
    held-open ``Live`` (its auto-refresh thread would catch the change anyway,
    but an explicit refresh keeps the box responsive between footer ticks).
    """
    card = getattr(pending, "card", None) if pending is not None else None
    if card is None:
        return
    if ui is not None:
        _ui_call(ui, "show_tool_card", card)
    elif getattr(pending, "live", None) is not None:
        pending.live.refresh()


def _dispatch_run_command_with_ui(
    args,
    config,
    history=None,
    usage_path=None,
    session_id: str | None = None,
    cancellation_token=None,
    retained_store=None,
    ui=None,
    interactive_help=None,
):
    prepared = prepare_run_command(args, config)
    if isinstance(prepared, str):
        if ui is not None:
            _ui_call(ui, "show_error", prepared)
        else:
            console.print(f"[red]{prepared}[/red]")
        return prepared

    allowed, denial = _agent_check_run_command(
        prepared,
        config,
        safety_history=history,
        usage_path=usage_path,
        session_id=session_id,
        cancellation_token=cancellation_token,
    )
    if not allowed:
        if ui is not None:
            _ui_call(ui, "show_notice", Text(denial, style="dim"))
        else:
            console.print(f"[dim]{denial}[/dim]")
        return denial

    display_mode = get_setting(config, "tool_call_display")
    metadata_text = f"model window {prepared.head_chars:,} / {prepared.tail_chars:,} chars"
    # One growing transcript box for the whole interactive session. Inline mode
    # holds a Rich ``Live`` open across turns so each model step appends in place
    # rather than re-printing a fresh box; heads-up re-renders the same card into
    # its live-tool slot. The card starts in its "running" state.
    card = InteractiveCommandCard(
        prepared.cmd,
        metadata_text,
        display_mode,
        time.perf_counter(),
    )
    live = None
    live_depth_cm = None

    try:
        if ui is not None:
            _ui_call(ui, "show_tool_card", card)
        else:
            live_depth_cm = track_live_display()
            live_depth_cm.__enter__()
            live = Live(
                card,
                refresh_per_second=8,
                console=console,
                auto_refresh=True,
                transient=False,
                vertical_overflow="crop",
            )
            live.start()
        process = InteractiveCommandProcess.start(prepared.cmd)
        unregister_cancel = (
            cancellation_token.register(process.kill_tree)
            if cancellation_token is not None else None
        )
        snapshot = process.wait_until_idle(
            check_in_seconds=_interactive_check_in_seconds(config),
            cancellation_token=cancellation_token,
        )
        card.seed_initial(snapshot)
        if ui is not None:
            _ui_call(ui, "show_tool_card", card)
        elif live is not None:
            live.refresh()
    except (KeyboardInterrupt, TurnCancelled):
        _stop_interactive_card_live(live, live_depth_cm)
        raise
    except Exception as e:
        _stop_interactive_card_live(live, live_depth_cm)
        return f"[error: {e}]"

    if snapshot.exited:
        from .orchestrator import format_run_command_output

        _stop_interactive_card_live(live, live_depth_cm)
        output, _output_id = format_run_command_output(
            snapshot.to_command_result(),
            prepared,
            retained_store,
        )
        if callable(unregister_cancel):
            unregister_cancel()
        return output

    pending = PendingRunCommand(
        process=process,
        prepared=prepared,
        call_id="",
        retained_store=retained_store,
        unregister_cancel=unregister_cancel,
        card=card,
        live=live,
        live_depth_cm=live_depth_cm,
    )
    include_help = True
    if interactive_help is not None:
        include_help = not interactive_help.get("sent", False)
        interactive_help["sent"] = True
    return RunCommandDispatchResult(
        _run_command_waiting_prompt(snapshot, include_help=include_help),
        pending,
    )


class _TurnRenderer:
    """Drives the live display and phase bookkeeping for one streamed turn.

    A :func:`run_agent` turn streams a model response that may interleave
    reasoning, assistant text, and tool calls. This object owns the transient
    Rich ``Live`` handles (the response-wait spinner, the tool-activity spinner,
    the inline streaming preview) and the per-turn phase state, and exposes the
    stream callbacks ``collect_stream_response`` expects. Folding that tangle of
    ``nonlocal`` flags into one named collaborator lets the turn loop read as a
    sequence of steps. One instance is reused across the loop's iterations and
    across stream retries; :meth:`begin_turn` resets the per-turn scratch state
    while leaving the live handles (which persist between turns) untouched.

    The turn *result* (final ``reply_text``/``tool_calls``/``reasoning_items``)
    is adopted from the stream collection via :meth:`adopt_stream_result`; the
    cancel/error checkpoints in ``run_agent`` read it back from here so a turn
    interrupted mid-stream still persists whatever text streamed so far.
    """

    def __init__(self, *, ui, interactive, status_items, metadata):
        self.ui = ui
        self.interactive = interactive
        self.status_items = status_items  # shared pending_status_history_items list
        self.metadata = metadata
        # Live handles persist across turns; stopped in run_agent's finally.
        self.spinner_live: Live | None = None
        self.stream_live: Live | None = None
        self.wait_indicator: ResponseWaitIndicator | None = None
        self.tool_indicator: ToolActivityIndicator | None = None
        self.thought_started = time.perf_counter()
        self.pending_interactive_command = None
        self._reset_turn_state()

    def _reset_turn_state(self) -> None:
        self.reply_text = ""
        self.tool_calls = []
        self.reasoning_items = []
        self.saw_reasoning = False
        self.got_text = False
        self.started_tool_positions: dict[str, int] = {}
        self.started_tool_names: list[str] = []
        self.tool_started_at: float | None = None
        self.tool_completed_at: float | None = None
        self.response_phase_completed = False
        self.tool_phase_completed = False
        self.stream_preview: StreamingMarkdownPreview | None = None

    def begin_turn(self, pending_interactive_command) -> None:
        self.pending_interactive_command = pending_interactive_command
        self._reset_turn_state()

    def adopt_stream_result(self, stream_result) -> None:
        self.reply_text = stream_result.reply_text
        self.tool_calls = stream_result.tool_calls
        self.reasoning_items = stream_result.reasoning_items
        self.saw_reasoning = stream_result.saw_reasoning
        self.got_text = stream_result.got_text

    # -- interactive-continuation invariant ---------------------------- #
    @property
    def interactive_continuation(self) -> bool:
        """True while a held-open interactive ``run_command`` owns the screen.

        This is the single rule the whole turn renderer keys off: during an
        interactive continuation the growing command card's own footer is the
        *only* live region, so every other per-turn chrome path -- the
        response-wait spinner, the tool-activity spinner, the streamed reply
        text, and the reasoning label -- must be suppressed. Inline, a second
        Rich ``Live`` would collide with the card's held-open ``Live``; in
        heads-up the chrome would leak as stray lines *outside* the card box
        (the "thinking/stdin outside the box" bug). Every method that would
        otherwise paint turn chrome checks this first.
        """
        return self.pending_interactive_command is not None

    def _animate_card_thinking(self) -> None:
        """Drive the card footer's "deciding next input" animation.

        Used wherever the non-interactive path would (re)open a response-wait
        spinner. The card replaces that spinner during a continuation, so we
        just nudge its footer instead of opening a competing ``Live``.
        """
        pending = self.pending_interactive_command
        if pending is not None and getattr(pending, "card", None) is not None:
            pending.card.set_thinking(self.thought_started)
            _refresh_interactive_card(self.ui, pending)

    # -- live handles -------------------------------------------------- #
    def _refresh_wait_indicator(self) -> None:
        if self.wait_indicator is not None and self.spinner_live is not None:
            self.spinner_live.update(self.wait_indicator, refresh=True)

    def _refresh_tool_indicator(self) -> None:
        if self.tool_indicator is not None and self.spinner_live is not None:
            self.spinner_live.update(self.tool_indicator, refresh=True)

    def stop_live(self) -> None:
        if self.spinner_live is not None:
            self.spinner_live.stop()
            self.spinner_live = None
        if self.stream_live is not None:
            self.stream_live.stop()
            self.stream_live = None

    def start_response_wait_now(self) -> None:
        """Show the response-wait spinner immediately (inline mode only).

        Called before the first stream so the spinner is visible during the
        potentially slow session prep; the UI-driven path starts its wait per
        turn via :meth:`ensure_response_wait`.
        """
        if self.ui is None:
            self.wait_indicator, self.spinner_live = _start_response_wait_indicator(
                self.interactive, self.thought_started
            )

    def ensure_response_wait(self) -> None:
        if self.spinner_live is not None:
            return
        self.thought_started = time.perf_counter()
        if self.interactive_continuation:
            self._animate_card_thinking()
            return
        if self.ui is not None:
            _ui_call(self.ui, "start_response_wait", self.thought_started)
        else:
            self.wait_indicator, self.spinner_live = _start_response_wait_indicator(
                self.interactive,
                self.thought_started,
            )

    # -- phase transitions --------------------------------------------- #
    def complete_response_phase(self) -> None:
        if self.response_phase_completed:
            return
        if self.spinner_live is not None:
            self.spinner_live.stop()
            self.spinner_live = None
        self.wait_indicator = None
        self.response_phase_completed = True
        if self.interactive_continuation:
            # The model's decision time is folded into the card's per-step
            # "·N.Ns" marker (added in the run_agent continuation block), so
            # there is no standalone "Decided next input" status line and nothing
            # is appended to status_items/history — the whole interaction stays
            # collapsed into the single run_command record.
            return
        status_text = response_start_status(
            time.perf_counter() - self.thought_started,
            has_reasoning=self.saw_reasoning,
        )
        self.status_items.append(
            status_history_item(status_text, "response", self.metadata)
        )
        if self.ui is not None:
            _ui_call(self.ui, "complete_response_phase", status_text)
            return
        if self.interactive:
            console.print(thought_complete_indicator(status_text))

    def complete_tool_phase(self) -> None:
        if self.tool_started_at is None or self.tool_phase_completed:
            return
        if self.spinner_live is not None:
            self.spinner_live.stop()
            self.spinner_live = None
        self.tool_phase_completed = True
        status_text = tool_activity_complete_status(
            (self.tool_completed_at or time.perf_counter()) - self.tool_started_at,
            tuple(self.started_tool_names),
        )
        self.status_items.append(
            status_history_item(status_text, "tool", self.metadata)
        )
        if self.ui is not None:
            _ui_call(self.ui, "complete_tool_phase", status_text)
            return
        if self.interactive:
            console.print(tool_complete_indicator(status_text))

    def note_provider_notice(self, text: str) -> None:
        """Surface a provider-reported condition (e.g. safety fallback) to the user."""
        if not text or self.interactive_continuation:
            return
        self.status_items.append(status_history_item(text, "notice", self.metadata))
        if self.ui is not None:
            _ui_call(self.ui, "show_notice", Text(f"⚠ {text}", style="yellow"))
        elif self.interactive:
            console.print(Text(f"⚠ {text}", style="yellow"))
        else:
            print(text, file=sys.stderr)

    def note_tool_call_started(self, item_id: str, call_id: str, name: str) -> None:
        if self.interactive_continuation:
            # Interactive continuations forbid tool calls; if the model emits one
            # anyway the turn loop nudges it back to stdin. Skip the tool-activity
            # spinner — a second Rich Live would collide with the card's Live.
            return
        keys = {value for value in (item_id, call_id) if value}
        positions = {
            self.started_tool_positions[key]
            for key in keys
            if key in self.started_tool_positions
        }
        if positions:
            position = min(positions)
            for key in keys:
                self.started_tool_positions[key] = position
            if name and not self.started_tool_names[position]:
                self.started_tool_names[position] = name
                if self.tool_indicator is not None:
                    self.tool_indicator.start_tool_call(str(position), name)
                    self._refresh_tool_indicator()
            return

        if self.tool_started_at is None:
            self.complete_response_phase()
            if self.stream_preview is not None:
                self.stream_preview.flush(refresh=False)
            if self.stream_live is not None:
                self.stream_live.stop()
                self.stream_live = None
            self.tool_started_at = time.perf_counter()
            if self.ui is not None:
                _ui_call(self.ui, "start_tool_activity", self.tool_started_at)
            elif self.interactive:
                self.tool_indicator = ToolActivityIndicator(self.tool_started_at)
                self.spinner_live = Live(
                    self.tool_indicator,
                    refresh_per_second=4,
                    console=console,
                    auto_refresh=True,
                    transient=True,
                )

        if not keys:
            keys = {f"tool_{len(self.started_tool_names)}"}
        position = len(self.started_tool_names)
        for key in keys:
            self.started_tool_positions[key] = position
        self.started_tool_names.append(name)
        if self.ui is not None:
            _ui_call(self.ui, "update_tool_activity", tuple(self.started_tool_names))
        elif self.tool_indicator is not None:
            self.tool_indicator.start_tool_call(str(position), name)
            if self.spinner_live is not None:
                self.spinner_live.start()
            self._refresh_tool_indicator()

    # -- stream callbacks ---------------------------------------------- #
    def on_stream_event(self, event, _result: StreamCollection) -> None:
        if isinstance(event, TextDelta):
            if not self.got_text:
                self.got_text = True
                self.complete_tool_phase()
                if self.interactive_continuation:
                    # Keep the "Deciding next input…" footer animating until the
                    # whole stream completes. The reply text is intentionally
                    # hidden for interactive continuations, so completing the
                    # response phase here at the first token would leave the rest
                    # of a long, often minutes-long interleaved-reasoning stream
                    # looking frozen — the gap the user sees after "Decided next
                    # input". The final complete_response_phase after the stream
                    # records the true turn duration.
                    pass
                else:
                    self.complete_response_phase()
                    if self.interactive:
                        # max_lines defaults to None so the crop bound is recomputed
                        # from the live terminal height on every paint — a mid-stream
                        # resize reflows instead of overflowing a frozen bound.
                        self.stream_live = InPlaceLive(
                            TailMarkdown(""),
                            console=console,
                            auto_refresh=False,
                            transient=True,
                            vertical_overflow="crop",
                        )
                        self.stream_live.start()
                        self.stream_preview = StreamingMarkdownPreview(self.stream_live)
                    elif self.ui is not None:
                        # New streamed message this turn; reset the UI's stream
                        # cursor so it appends in order rather than overwriting the
                        # previous turn's bubble above the intervening tool cards.
                        _ui_call(self.ui, "begin_assistant_message")
            if self.interactive_continuation:
                # Reply text is the stdin line; it surfaces as the card's
                # "stdin> …" marker, never as a standalone streamed message.
                self.reply_text += event.delta
            elif self.stream_preview is not None:
                self.stream_preview.append(event.delta)
            elif self.ui is not None:
                _ui_call(self.ui, "append_stream_delta", event.delta)
            else:
                self.reply_text += event.delta
        elif isinstance(event, ToolCallStarted):
            self.note_tool_call_started(event.id, event.call_id, event.name)
        elif isinstance(event, ToolCallDone):
            self.note_tool_call_started(event.id, event.call_id, event.name)
            self.tool_completed_at = time.perf_counter()
        elif isinstance(event, (ReasoningStarted, ReasoningDone)):
            self.saw_reasoning = True
            self._note_reasoning()

    def _note_reasoning(self) -> None:
        """Surface that the model is reasoning on the active wait indicator.

        Suppressed during an interactive continuation: there is no response-wait
        status then (the card footer stands in for it), so pushing a reasoning
        label would materialise a stray "Thinking…" line outside the card box.
        """
        if self.interactive_continuation:
            return
        if self.ui is not None:
            _ui_call(self.ui, "set_response_wait_has_reasoning", True)
        elif self.wait_indicator is not None:
            self.wait_indicator.has_reasoning = True
            self._refresh_wait_indicator()

    def on_stream_attempt_end(self, result: StreamCollection, _retry_stream: bool) -> None:
        if not self.interactive_continuation:
            # Skipped during a continuation: the reply text is the stdin line,
            # shown in the card's "stdin> …" marker, so replaying it here as a
            # standalone message is what leaked "go library"/"quit" outside the
            # card box.
            if result.final_text and self.stream_preview is not None:
                self.stream_preview.replace(result.final_text)
            elif result.final_text and self.ui is not None:
                _ui_call(self.ui, "replace_stream_text", result.final_text)
        if self.stream_preview is not None:
            result.reply_text = self.stream_preview.text
            # Paint the authoritative final text into the live region before it
            # is stopped, so the last visible streamed frame matches the reprint
            # rather than freezing on a stale, throttled tail.
            self.stream_preview.flush(refresh=True)
        self.reply_text = result.reply_text
        if self.spinner_live is not None:
            self.spinner_live.stop()
            self.spinner_live = None
        if self.stream_live is not None:
            self.stream_live.stop()
            self.stream_live = None

    def on_stream_retry(self) -> None:
        self._reset_turn_state()
        self.status_items.clear()
        self.tool_indicator = None
        self.wait_indicator = None
        self.thought_started = time.perf_counter()
        if self.interactive_continuation:
            # Same rule as ensure_response_wait: the card footer is the only live
            # region during an interactive continuation, so keep the "deciding"
            # animation going rather than opening a competing response-wait Live.
            self._animate_card_thinking()
            return
        if self.ui is not None:
            _ui_call(self.ui, "retry_stream")
            _ui_call(self.ui, "start_response_wait", self.thought_started)
        else:
            self.wait_indicator, self.spinner_live = _start_response_wait_indicator(
                self.interactive,
                self.thought_started,
            )


class TurnCheckpointer:
    """Persists partial turn state when a run errors or is cancelled.

    Reads the always-synced ``persistence.history`` / ``renderer.metadata``
    instead of closing over run-locals that are rebound during session prep.
    ``active_tool_call`` mirrors the tool currently executing so a cancel can
    distinguish "may have made partial changes" from "never ran".
    """

    def __init__(self, *, persistence: SessionPersistence, renderer: _TurnRenderer,
                 status_items: list[dict]):
        self.persistence = persistence
        self.renderer = renderer
        self.status_items = status_items  # shared pending_status_history_items list
        self.active_tool_call = None

    def flush_status_items(self) -> None:
        if self.status_items:
            self.persistence.history.extend(self.status_items)
            self.status_items.clear()

    def flush_error_state(self) -> None:
        self.flush_status_items()
        history = self.persistence.history
        renderer = self.renderer
        if renderer.reply_text and not renderer.tool_calls:
            history.append(
                {"role": "assistant", "content": renderer.reply_text, **renderer.metadata}
            )
        self.persistence.save()

    def checkpoint_cancelled_turn(self) -> None:
        self.flush_status_items()
        history = self.persistence.history
        renderer = self.renderer
        metadata = renderer.metadata

        recorded_reasoning_ids = {
            str(item.get("id"))
            for item in history
            if isinstance(item, dict) and item.get("type") == "reasoning"
        }
        for item in renderer.reasoning_items:
            if str(item.id) not in recorded_reasoning_ids:
                stored_reasoning = {
                    "type": "reasoning",
                    "id": item.id,
                    "summary": [],
                    **metadata,
                }
                if item.provider_content:
                    stored_reasoning["provider_content"] = item.provider_content
                history.append(stored_reasoning)

        if renderer.reply_text:
            history.append({"role": "assistant", "content": renderer.reply_text, **metadata})

        recorded_call_ids = {
            str(item.get("call_id"))
            for item in history
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        active_call_id = (
            str(self.active_tool_call.call_id)
            if self.active_tool_call is not None else None
        )
        for item in renderer.tool_calls:
            if str(item.call_id) in recorded_call_ids:
                continue
            if str(item.call_id) == active_call_id:
                output = "[cancelled by user; execution may have made partial changes]"
            else:
                output = "[cancelled by user before execution]"
            stored_call = {
                "type": "function_call",
                "id": item.id,
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
                **metadata,
            }
            if item.provider_content:
                stored_call["provider_content"] = item.provider_content
            history.extend([
                stored_call,
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output,
                    **metadata,
                },
            ])

        history.append({
            "role": "assistant",
            "content": "[Turn cancelled by user.]",
            **metadata,
        })
        self.persistence.save_turn()


def _advance_interactive_continuation(
    pending: PendingRunCommand,
    renderer: _TurnRenderer,
    input_items: list,
    *,
    config: dict,
    cancellation_token: CancellationToken,
    retained_store,
    ui,
    interactive_help: dict,
) -> tuple[list, "PendingRunCommand | None"]:
    """One model->stdin->process round of a held-open interactive command.

    Returns ``(new input items, still-pending command or None)``. The model's
    control reply and the next prompt stay in the live conversation only;
    history keeps just the one collapsed run_command record.
    """
    if renderer.tool_calls:
        # A stray mid-interaction tool call: remind and re-prompt without
        # touching the process.
        return (
            input_items + [{
                "role": "user",
                "content": _interactive_tool_call_reminder(),
            }],
            pending,
        )
    # Capture the decision time now (before ``_continue`` adds the process
    # wait) so it lands on this step's "·N.Ns" marker.
    decision_seconds = max(0.0, time.perf_counter() - renderer.thought_started)
    terminal_reply = renderer.reply_text
    input_items = input_items + [{
        "role": "assistant",
        "content": terminal_reply,
    }]
    snapshot, terminal_action, terminal_kind = _continue_interactive_command(
        pending,
        terminal_reply,
        config=config,
        cancellation_token=cancellation_token,
    )
    _record_interactive_input(pending, terminal_kind, terminal_action)
    # Grow the single open card in place instead of printing a fresh box:
    # append this step's stdin marker + delta output and let the footer
    # resolve to the new waiting/exit state.
    pending.card.add_step(
        _interaction_marker_text(terminal_kind, terminal_action),
        snapshot.to_delta_command_result().full_model_output(),
        decision_seconds,
        exited=snapshot.exited,
        exit_code=snapshot.exit_code if snapshot.exited else None,
        check_in=bool(getattr(snapshot, "check_in", False)),
    )
    _refresh_interactive_card(ui, pending)
    if snapshot.exited:
        _close_interactive_card_live(pending)
        final_output = _format_finished_interactive_output(
            pending,
            snapshot,
            retained_store,
        )
        input_items = input_items + [{
            "role": "user",
            "content": _run_command_final_prompt(final_output),
        }]
        _finalize_interactive_record(pending, final_output)
        return input_items, None
    include_help = not interactive_help["sent"]
    interactive_help["sent"] = True
    waiting_prompt = _run_command_waiting_prompt(snapshot, include_help=include_help)
    return (
        input_items + [{
            "role": "user",
            "content": waiting_prompt,
        }],
        pending,
    )


def _build_tool_hooks(
    *,
    config: dict,
    history: list,
    usage_path,
    session_id: str,
    cancellation_token: CancellationToken,
    retained_store,
    ui,
    interactive_help: dict,
    root_node,
    artifact_store,
    client,
) -> ToolExecutionHooks:
    """Wire the tool-execution callbacks for a run.

    The dispatch functions are referenced as module globals at call time so
    tests patching ``jarv.agent._dispatch_*`` keep intercepting.
    """
    import json

    from .session_render import tool_call_card, tool_call_card_from_args
    from .tool_outputs import summarize_tool_output

    def _on_tool_error(message: str) -> None:
        if ui is not None:
            _ui_call(ui, "show_error", message)
        else:
            console.print(f"[red]{message}[/red]")

    def _on_parallel_read(_item, read_args: dict) -> None:
        _print_tool_card(
            tool_call_card_from_args(
                "read",
                read_args,
                display_mode=get_setting(config, "tool_call_display"),
            ),
            config,
            ui=ui,
        )

    def _on_parallel_web_search(_item, search_args: dict) -> None:
        _print_tool_card(
            tool_call_card_from_args(
                "web_search",
                search_args,
                display_mode=get_setting(config, "tool_call_display"),
            ),
            config,
            ui=ui,
        )

    def _run_edit(edit_args: dict) -> str:
        output = dispatch_edit_tool(
            edit_args,
            config=config,
            cancellation_token=cancellation_token,
        )
        # Card prints after dispatch so any confirmation panel appears first
        # and the card reflects done/failed status.
        _print_tool_card(
            tool_call_card(
                {"name": "edit", "arguments": json.dumps(edit_args, ensure_ascii=True)},
                summarize_tool_output(output),
                display_mode=get_setting(config, "tool_call_display"),
            ),
            config,
            ui=ui,
        )
        return output

    return ToolExecutionHooks(
        on_parallel_read=_on_parallel_read,
        on_parallel_web_search=_on_parallel_web_search,
        run_edit=_run_edit,
        run_command=lambda args: _dispatch_run_command_with_ui(
            args,
            config,
            history,
            usage_path=usage_path,
            session_id=session_id,
            cancellation_token=cancellation_token,
            retained_store=retained_store,
            ui=ui,
            interactive_help=interactive_help,
        ),
        run_spawn=lambda args: _dispatch_spawn_with_ui(
            args,
            root_node,
            artifact_store,
            client,
            config,
            cancellation_token=cancellation_token,
            retained_store=retained_store,
            ui=ui,
        ),
        run_ask_user=lambda args: _dispatch_ask_user(args, config, ui=ui),
        on_tool_error=_on_tool_error,
    )


def build_instructions(config: dict) -> str:
    instructions = config["system_prompt"] + f"\n\nSystem info:\n{get_system_info()}"
    project_context = build_project_context(config)
    if project_context:
        instructions += "\n\n" + project_context
    return instructions


def run_agent(
    query: str,
    config: dict,
    client=None,
    propagate_keyboard_interrupt: bool = False,
    new_session: bool = False,
    incognito: bool = False,
    heads_up: bool = False,
    ui=None,
) -> AgentRunResult:
    config = dict(config)
    config["tool_call_display"] = resolve_tool_call_display(
        config,
        heads_up=heads_up,
    )
    interactive = sys.stdout.isatty() and ui is None
    history: list = []
    metadata: dict = {}
    session_context = None
    artifact_file = None
    artifact_store = None
    reads_file = None
    retained_store = None
    usage_path = None
    persistence = SessionPersistence(incognito=incognito)
    persistence.history = history
    pending_status_history_items: list[dict] = []
    # The terminal-control help line is sent on the first interactive waiting
    # prompt of the session only; later prompts carry just state.
    interactive_help = {"sent": False}
    cancellation_token = CancellationToken()
    renderer = _TurnRenderer(
        ui=ui,
        interactive=interactive,
        status_items=pending_status_history_items,
        metadata=metadata,
    )
    checkpointer = TurnCheckpointer(
        persistence=persistence,
        renderer=renderer,
        status_items=pending_status_history_items,
    )
    _ui_call(ui, "bind_cancel_token", cancellation_token)
    _ui_call(ui, "start_turn", query, config)
    renderer.start_response_wait_now()

    sigint_cancel_scope = cancel_token_on_sigint(cancellation_token)
    try:
        sigint_cancel_scope.__enter__()
        if client is None:
            client = create_client(config)

        if new_session:
            forget_current_session()
        session_context = prepare_session_context(
            mark_message=not incognito,
            persist_metadata=not incognito,
        )
        persistence.session_context = session_context
        history = [] if (new_session or incognito) else load_history(session_context.history_file)
        persistence.history = history
        web_search_read_nudge_sent = history_has_web_search_read_nudge(history)
        metadata = history_metadata(session_context)
        renderer.metadata = metadata

        artifact_file = artifact_file_for(session_context.history_file)
        artifact_store = load_artifact_store(artifact_file)
        persistence.artifact_file = artifact_file
        persistence.artifact_store = artifact_store
        reads_file = reads_file_for(session_context.history_file)
        retained_store = load_retained_output_store(reads_file)
        persistence.reads_file = reads_file
        persistence.retained_store = retained_store
        usage_path = usage_file_for(session_context.history_file)
        root_node = AgentNode(
            label="root",
            depth=0,
            parent_label=None,
            task=query,
            sterile=False,
            visible_labels=artifact_store.all_labels(),
            usage_path=usage_path,
            session_id=session_context.session_id,
            incognito=incognito,
        )

        history.append({"role": "user", "content": query, "id": new_frame_id(), **metadata})

        instructions = build_instructions(config)
        tools = build_agent_tools(config)
        input_items = build_input(
            history,
            model=config["model"],
            config=config,
            instructions=instructions,
            tools=tools,
        )

        kwargs = dict(
            model=config["model"],
            instructions=instructions,
            tools=tools,
            input=input_items,
        )
        effort = config.get("reasoning_effort")
        if effort:
            kwargs["reasoning"] = {"effort": effort}

        pending_interactive_command: PendingRunCommand | None = None

        while True:
            renderer.begin_turn(pending_interactive_command)

            renderer.ensure_response_wait()

            # Keep the full tool list on every turn, including interactive
            # continuations. Blanking it there changed the cached prompt prefix
            # (instructions -> tools -> input), forcing a full re-prefill at the
            # enter/exit boundaries, and a tool-trained model handed no tools
            # spills tool-shaped text into stdin. Stray mid-interaction tool
            # calls are still caught and dropped by the reminder branch below
            # (which `continue`s before the execution path), so passing the real
            # tools never risks a nested run_command.
            active_tools = kwargs.get("tools", [])

            _ctx_breakdown = estimate_context_breakdown(
                config["model"],
                kwargs.get("instructions", ""),
                active_tools,
                kwargs.get("input", []),
            )

            def make_stream():
                return stream_response(
                    client, config,
                    kwargs["model"], kwargs["instructions"],
                    active_tools, kwargs["input"],
                    reasoning=kwargs.get("reasoning"),
                    prompt_cache_key=f"jarv:{session_context.session_id}",
                    cancellation_token=cancellation_token,
                )

            stream_result = collect_stream_response(
                make_stream,
                on_event=renderer.on_stream_event,
                on_attempt_end=renderer.on_stream_attempt_end,
                on_retry=renderer.on_stream_retry,
            )
            renderer.adopt_stream_result(stream_result)
            final_response = stream_result.final_response
            if not incognito:
                from .provider_catalog import configured_service_tier

                record_response_usage(
                    usage_path,
                    session_context.session_id,
                    config["model"],
                    final_response,
                    "root",
                    provider=str(config.get("provider") or "openai"),
                    requested_service_tier=configured_service_tier(config),
                    context_breakdown=_ctx_breakdown,
                    output_text=stream_usage_output_text(
                        renderer.reply_text, renderer.tool_calls
                    ),
                )
            renderer.complete_tool_phase()
            renderer.complete_response_phase()
            fallback_note = provider_response_notice(config, final_response)
            if fallback_note:
                renderer.note_provider_notice(fallback_note)
            if renderer.got_text:
                if pending_interactive_command is not None:
                    pass
                elif ui is not None:
                    _ui_call(ui, "finish_assistant_message", renderer.reply_text)
                elif interactive:
                    console.print(Markdown(flatten_headings(renderer.reply_text)))
                else:
                    print(renderer.reply_text)

            if pending_interactive_command is not None:
                checkpointer.flush_status_items()
                kwargs["input"], pending_interactive_command = (
                    _advance_interactive_continuation(
                        pending_interactive_command,
                        renderer,
                        kwargs["input"],
                        config=config,
                        cancellation_token=cancellation_token,
                        retained_store=retained_store,
                        ui=ui,
                        interactive_help=interactive_help,
                    )
                )
                continue

            if renderer.tool_calls:
                print_mode_spacer(config)
                checkpointer.flush_status_items()

                tool_hooks = _build_tool_hooks(
                    config=config,
                    history=history,
                    usage_path=usage_path,
                    session_id=session_context.session_id,
                    cancellation_token=cancellation_token,
                    retained_store=retained_store,
                    ui=ui,
                    interactive_help=interactive_help,
                    root_node=root_node,
                    artifact_store=artifact_store,
                    client=client,
                )

                checkpointer.active_tool_call = (
                    renderer.tool_calls[0] if renderer.tool_calls else None
                )

                def _execute_tools(_new_input, append_tool_result):
                    nonlocal web_search_read_nudge_sent
                    checkpointer.active_tool_call = (
                        renderer.tool_calls[0] if renderer.tool_calls else None
                    )
                    result = execute_tool_calls(
                        renderer.tool_calls,
                        node=root_node,
                        store=artifact_store,
                        client=client,
                        config=config,
                        append_tool_result=append_tool_result,
                        hooks=tool_hooks,
                        cancellation_token=cancellation_token,
                        retained_store=retained_store,
                        web_search_read_nudge_sent=web_search_read_nudge_sent,
                    )
                    web_search_read_nudge_sent = result.web_search_read_nudge_sent
                    checkpointer.active_tool_call = None
                    return result

                kwargs["input"], _exec_result = run_tool_execution_round(
                    kwargs["input"],
                    stream_result,
                    model=config["model"],
                    config=config,
                    instructions=kwargs["instructions"],
                    tools=kwargs["tools"],
                    reasoning_kwargs={"history": history, "metadata": metadata},
                    tool_result_kwargs={"history": history, "metadata": metadata},
                    execute_tool_calls_fn=_execute_tools,
                )
                pending_interactive_command = _exec_result.pending_command
                if pending_interactive_command is not None:
                    _attach_interactive_output_item(
                        pending_interactive_command, history
                    )
            else:
                checkpointer.flush_status_items()
                history.append({"role": "assistant", "content": renderer.reply_text, **metadata})
                persistence.save_turn()
                _print_agent_usage_if_enabled(
                    config,
                    usage_path,
                    session_context.session_id,
                    ui=ui,
                    heads_up=heads_up,
                )
                break
        return AgentRunResult()
    except (KeyboardInterrupt, TurnCancelled):
        cancellation_token.cancel()
        pending = locals().get("pending_interactive_command")
        if pending is not None:
            # Close the held-open card Live before the checkpoint prints below,
            # so its final frame isn't corrupted by intervening console output.
            _close_interactive_card_live(pending)
            try:
                pending.process.kill_tree()
            except Exception:
                pass
            if callable(getattr(pending, "unregister_cancel", None)):
                pending.unregister_cancel()
        if session_context is not None:
            checkpointer.checkpoint_cancelled_turn()
        return AgentRunResult(cancelled=True, prompt=query)
    except (ProviderError, Exception) as e:
        _close_interactive_card_live(locals().get("pending_interactive_command"))
        label = "API error" if isinstance(e, ProviderError) else "Unexpected error"
        message = f"{label}: {e}"
        if ui is not None:
            _ui_call(ui, "show_error", message)
        else:
            style = "red"
            console.print(f"[{style}]{label}:[/{style}] {escape(str(e))}")
        checkpointer.flush_error_state()
        return AgentRunResult(error=str(e))
    finally:
        sigint_cancel_scope.__exit__(None, None, None)
        _ui_call(ui, "unbind_cancel_token")
        renderer.stop_live()
        # Safety net: close any interactive card Live the paths above missed.
        _close_interactive_card_live(locals().get("pending_interactive_command"))


