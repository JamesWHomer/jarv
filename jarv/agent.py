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
    terminal_size,
    track_live_display,
)
from .history import (
    artifact_file_for,
    forget_current_session,
    get_shell_name,
    history_metadata,
    load_history,
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
    ResponseWaitIndicator,
    RunningCommandCard,
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
    response_start_status,
    response_wait_label,
    tool_activity_complete_status,
    tool_activity_label,
    tool_complete_indicator,
    thought_complete_indicator,
)
from .context_budget import build_input
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
    _continue_interactive_command,
    _first_terminal_action,
    _format_elapsed_seconds,
    _format_finished_interactive_output,
    _interactive_check_in_seconds,
    _interactive_command_card,
    _parse_terminal_control,
    _run_command_final_prompt,
    _run_command_waiting_prompt,
    _show_interactive_command_card,
    _terminal_action_display,
)


def _dispatch_run_command_with_ui(
    args,
    config,
    history=None,
    usage_path=None,
    session_id: str | None = None,
    cancellation_token=None,
    retained_store=None,
    ui=None,
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
    running_card = RunningCommandCard(
        prepared.cmd,
        metadata_text,
        display_mode,
        time.perf_counter(),
    )

    try:
        if ui is not None:
            _ui_call(ui, "show_tool_card", running_card)
            process = InteractiveCommandProcess.start(prepared.cmd)
            unregister_cancel = (
                cancellation_token.register(process.kill_tree)
                if cancellation_token is not None else None
            )
            snapshot = process.wait_until_idle(
                check_in_seconds=_interactive_check_in_seconds(config),
                cancellation_token=cancellation_token,
            )
        else:
            with track_live_display(), Live(
                running_card,
                refresh_per_second=4,
                console=console,
                auto_refresh=True,
                transient=False,
            ) as live:
                process = InteractiveCommandProcess.start(prepared.cmd)
                unregister_cancel = (
                    cancellation_token.register(process.kill_tree)
                    if cancellation_token is not None else None
                )
                snapshot = process.wait_until_idle(
                    check_in_seconds=_interactive_check_in_seconds(config),
                    cancellation_token=cancellation_token,
                )
                live.update(
                    _interactive_command_card(
                        prepared,
                        snapshot,
                        config,
                        status="done" if snapshot.exited else "waiting",
                    ),
                    refresh=True,
                )
    except (KeyboardInterrupt, TurnCancelled):
        raise
    except Exception as e:
        return f"[error: {e}]"

    if ui is not None:
        _ui_call(
            ui,
            "show_tool_card",
            _interactive_command_card(
                prepared,
                snapshot,
                config,
                status="done" if snapshot.exited else "waiting",
            ),
        )

    if snapshot.exited:
        from .orchestrator import format_run_command_output

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
    )
    return RunCommandDispatchResult(
        _run_command_waiting_prompt(snapshot),
        pending,
    )


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
    reply_text = ""
    tool_calls = []
    history: list = []
    metadata: dict = {}
    session_context = None
    artifact_file = None
    artifact_store = None
    reads_file = None
    retained_store = None
    usage_path = None
    persistence = SessionPersistence(incognito=incognito)
    wait_indicator: ResponseWaitIndicator | None = None
    tool_indicator: ToolActivityIndicator | None = None
    spinner_live: Live | None = None
    stream_live: Live | None = None
    reasoning_items = []
    active_tool_call = None
    pending_status_history_items: list[dict] = []
    cancellation_token = CancellationToken()
    thought_started = time.perf_counter()
    _ui_call(ui, "bind_cancel_token", cancellation_token)
    _ui_call(ui, "start_turn", query, config)
    if ui is None:
        wait_indicator, spinner_live = _start_response_wait_indicator(interactive, thought_started)

    def _refresh_wait_indicator() -> None:
        if wait_indicator is not None and spinner_live is not None:
            spinner_live.update(wait_indicator, refresh=True)

    def _refresh_tool_indicator() -> None:
        if tool_indicator is not None and spinner_live is not None:
            spinner_live.update(tool_indicator, refresh=True)

    def _stop_live_displays() -> None:
        nonlocal spinner_live, stream_live
        if spinner_live is not None:
            spinner_live.stop()
            spinner_live = None
        if stream_live is not None:
            stream_live.stop()
            stream_live = None

    def _flush_error_state() -> None:
        if pending_status_history_items:
            history.extend(pending_status_history_items)
            pending_status_history_items.clear()
        if reply_text and not tool_calls:
            history.append({"role": "assistant", "content": reply_text, **metadata})
        persistence.save()

    def _checkpoint_cancelled_turn() -> None:
        if pending_status_history_items:
            history.extend(pending_status_history_items)
            pending_status_history_items.clear()

        recorded_reasoning_ids = {
            str(item.get("id"))
            for item in history
            if isinstance(item, dict) and item.get("type") == "reasoning"
        }
        for item in reasoning_items:
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

        if reply_text:
            history.append({"role": "assistant", "content": reply_text, **metadata})

        recorded_call_ids = {
            str(item.get("call_id"))
            for item in history
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        active_call_id = (
            str(active_tool_call.call_id)
            if active_tool_call is not None else None
        )
        for item in tool_calls:
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
        persistence.save_turn()

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

        history.append({"role": "user", "content": query, **metadata})

        instructions = (
            config["system_prompt"]
            + f"\n\nSystem info:\n{get_system_info()}"
        )
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
            reply_text = ""
            tool_calls = []
            reasoning_items = []
            saw_reasoning = False
            got_text = False
            started_tool_positions: dict[str, int] = {}
            started_tool_names: list[str] = []
            tool_started_at: float | None = None
            tool_completed_at: float | None = None
            response_phase_completed = False
            tool_phase_completed = False
            stream_preview: StreamingMarkdownPreview | None = None

            def complete_response_phase() -> None:
                nonlocal spinner_live, wait_indicator, response_phase_completed
                if response_phase_completed:
                    return
                if spinner_live is not None:
                    spinner_live.stop()
                    spinner_live = None
                wait_indicator = None
                response_phase_completed = True
                if pending_interactive_command is not None:
                    return
                status_text = response_start_status(
                    time.perf_counter() - thought_started,
                    has_reasoning=saw_reasoning,
                )
                pending_status_history_items.append(
                    status_history_item(status_text, "response", metadata)
                )
                if ui is not None:
                    _ui_call(ui, "complete_response_phase", status_text)
                    return
                if interactive:
                    console.print(thought_complete_indicator(status_text))

            def complete_tool_phase() -> None:
                nonlocal spinner_live, tool_phase_completed
                if tool_started_at is None or tool_phase_completed:
                    return
                if spinner_live is not None:
                    spinner_live.stop()
                    spinner_live = None
                tool_phase_completed = True
                status_text = tool_activity_complete_status(
                    (tool_completed_at or time.perf_counter())
                    - tool_started_at,
                    tuple(started_tool_names),
                )
                pending_status_history_items.append(
                    status_history_item(status_text, "tool", metadata)
                )
                if ui is not None:
                    _ui_call(ui, "complete_tool_phase", status_text)
                    return
                if interactive:
                    console.print(tool_complete_indicator(status_text))

            def append_status_history_items() -> None:
                if pending_status_history_items:
                    history.extend(pending_status_history_items)
                    pending_status_history_items.clear()

            def note_tool_call_started(
                item_id: str,
                call_id: str,
                name: str,
            ) -> None:
                nonlocal spinner_live, stream_live, tool_indicator, tool_started_at
                keys = {value for value in (item_id, call_id) if value}
                positions = {
                    started_tool_positions[key]
                    for key in keys
                    if key in started_tool_positions
                }
                if positions:
                    position = min(positions)
                    for key in keys:
                        started_tool_positions[key] = position
                    if name and not started_tool_names[position]:
                        started_tool_names[position] = name
                        if tool_indicator is not None:
                            tool_indicator.start_tool_call(str(position), name)
                            _refresh_tool_indicator()
                    return

                if tool_started_at is None:
                    complete_response_phase()
                    if stream_preview is not None:
                        stream_preview.flush(refresh=False)
                    if stream_live is not None:
                        stream_live.stop()
                        stream_live = None
                    tool_started_at = time.perf_counter()
                    if ui is not None:
                        _ui_call(ui, "start_tool_activity", tool_started_at)
                    elif interactive:
                        tool_indicator = ToolActivityIndicator(tool_started_at)
                        spinner_live = Live(
                            tool_indicator,
                            refresh_per_second=4,
                            console=console,
                            auto_refresh=True,
                            transient=True,
                        )

                if not keys:
                    keys = {f"tool_{len(started_tool_names)}"}
                position = len(started_tool_names)
                for key in keys:
                    started_tool_positions[key] = position
                started_tool_names.append(name)
                if ui is not None:
                    _ui_call(ui, "update_tool_activity", tuple(started_tool_names))
                elif tool_indicator is not None:
                    tool_indicator.start_tool_call(str(position), name)
                    if spinner_live is not None:
                        spinner_live.start()
                    _refresh_tool_indicator()

            if spinner_live is None:
                thought_started = time.perf_counter()
                if ui is not None:
                    _ui_call(ui, "start_response_wait", thought_started)
                else:
                    wait_indicator, spinner_live = _start_response_wait_indicator(interactive, thought_started)

            active_tools = [] if pending_interactive_command is not None else kwargs.get("tools", [])

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

            def on_stream_event(event, _result: StreamCollection) -> None:
                nonlocal reply_text, got_text, saw_reasoning, stream_live, stream_preview
                nonlocal tool_completed_at
                if isinstance(event, TextDelta):
                    if not got_text:
                        got_text = True
                        complete_tool_phase()
                        complete_response_phase()
                        if pending_interactive_command is not None:
                            pass
                        elif interactive:
                            _, term_h = terminal_size(console=console)
                            stream_max_lines = term_h - 2
                            stream_live = InPlaceLive(
                                TailMarkdown("", stream_max_lines),
                                console=console,
                                auto_refresh=False,
                                transient=True,
                                vertical_overflow="crop",
                            )
                            stream_live.start()
                            stream_preview = StreamingMarkdownPreview(
                                stream_live,
                                stream_max_lines,
                            )
                    if pending_interactive_command is not None:
                        reply_text += event.delta
                    elif stream_preview is not None:
                        stream_preview.append(event.delta)
                    elif ui is not None:
                        _ui_call(ui, "append_stream_delta", event.delta)
                    else:
                        reply_text += event.delta
                elif isinstance(event, ToolCallStarted):
                    note_tool_call_started(
                        event.id,
                        event.call_id,
                        event.name,
                    )
                elif isinstance(event, ToolCallDone):
                    note_tool_call_started(
                        event.id,
                        event.call_id,
                        event.name,
                    )
                    tool_completed_at = time.perf_counter()
                elif isinstance(event, ReasoningStarted):
                    saw_reasoning = True
                    if ui is not None:
                        _ui_call(ui, "set_response_wait_has_reasoning", True)
                    elif wait_indicator is not None:
                        wait_indicator.has_reasoning = True
                        _refresh_wait_indicator()
                elif isinstance(event, ReasoningDone):
                    saw_reasoning = True
                    if ui is not None:
                        _ui_call(ui, "set_response_wait_has_reasoning", True)
                    elif wait_indicator is not None:
                        wait_indicator.has_reasoning = True
                        _refresh_wait_indicator()

            def on_stream_attempt_end(
                result: StreamCollection,
                _retry_stream: bool,
            ) -> None:
                nonlocal reply_text, spinner_live, stream_live, stream_preview
                if result.final_text and stream_preview is not None:
                    stream_preview.replace(result.final_text)
                elif result.final_text and ui is not None:
                    _ui_call(ui, "replace_stream_text", result.final_text)
                if stream_preview is not None:
                    result.reply_text = stream_preview.text
                    stream_preview.flush(refresh=False)
                reply_text = result.reply_text
                if spinner_live is not None:
                    spinner_live.stop()
                    spinner_live = None
                if stream_live is not None:
                    stream_live.stop()
                    stream_live = None

            def on_stream_retry() -> None:
                nonlocal reply_text, tool_calls, reasoning_items, saw_reasoning
                nonlocal got_text, started_tool_positions, started_tool_names
                nonlocal tool_started_at, tool_completed_at, spinner_live
                nonlocal response_phase_completed, tool_phase_completed
                nonlocal stream_preview, tool_indicator, wait_indicator, thought_started
                reply_text = ""
                tool_calls = []
                reasoning_items = []
                saw_reasoning = False
                got_text = False
                started_tool_positions = {}
                started_tool_names = []
                tool_started_at = None
                tool_completed_at = None
                response_phase_completed = False
                tool_phase_completed = False
                pending_status_history_items.clear()
                stream_preview = None
                tool_indicator = None
                wait_indicator = None
                thought_started = time.perf_counter()
                if ui is not None:
                    _ui_call(ui, "retry_stream")
                    _ui_call(ui, "start_response_wait", thought_started)
                else:
                    wait_indicator, spinner_live = _start_response_wait_indicator(
                        interactive,
                        thought_started,
                    )

            stream_result = collect_stream_response(
                make_stream,
                on_event=on_stream_event,
                on_attempt_end=on_stream_attempt_end,
                on_retry=on_stream_retry,
            )
            reply_text = stream_result.reply_text
            tool_calls = stream_result.tool_calls
            reasoning_items = stream_result.reasoning_items
            saw_reasoning = stream_result.saw_reasoning
            got_text = stream_result.got_text
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
                    output_text=stream_usage_output_text(reply_text, tool_calls),
                )
            complete_tool_phase()
            complete_response_phase()
            if got_text:
                if pending_interactive_command is not None:
                    pass
                elif ui is not None:
                    _ui_call(ui, "finish_assistant_message", reply_text)
                elif interactive:
                    console.print(Markdown(flatten_headings(reply_text)))
                else:
                    print(reply_text)

            if pending_interactive_command is not None:
                append_status_history_items()
                if tool_calls:
                    kwargs["input"] = kwargs["input"] + [{
                        "role": "user",
                        "content": (
                            "A terminal command is currently waiting for input. "
                            "Do not call tools. Reply only with stdin text or one of "
                            "<ENTER>, <WAIT>, <WAIT 10s>, <CTRL_C>, <EOF>, "
                            "<CTRL_D>, <ESC>, <TAB>, <UP>, <DOWN>, <LEFT>, "
                            "<RIGHT>."
                        ),
                    }]
                    continue
                terminal_reply = reply_text
                kwargs["input"] = kwargs["input"] + [{
                    "role": "assistant",
                    "content": terminal_reply,
                }]
                snapshot, terminal_action = _continue_interactive_command(
                    pending_interactive_command,
                    terminal_reply,
                    config=config,
                    cancellation_token=cancellation_token,
                )
                history.append({
                    "role": "assistant",
                    "content": (
                        "[terminal input sent]\n"
                        "```text\n"
                        f"{terminal_action}\n"
                        "```"
                    ),
                    **metadata,
                })
                _show_interactive_command_card(
                    pending_interactive_command,
                    snapshot,
                    config,
                    status="done" if snapshot.exited else "waiting",
                    terminal_reply=terminal_action,
                    ui=ui,
                )
                if snapshot.exited:
                    final_output = _format_finished_interactive_output(
                        pending_interactive_command,
                        snapshot,
                        retained_store,
                    )
                    kwargs["input"] = kwargs["input"] + [{
                        "role": "user",
                        "content": _run_command_final_prompt(final_output),
                    }]
                    history.append({
                        "role": "user",
                        "content": _run_command_final_prompt(final_output),
                        **metadata,
                    })
                    pending_interactive_command = None
                else:
                    waiting_prompt = _run_command_waiting_prompt(snapshot)
                    kwargs["input"] = kwargs["input"] + [{
                        "role": "user",
                        "content": waiting_prompt,
                    }]
                    history.append({
                        "role": "user",
                        "content": waiting_prompt,
                        **metadata,
                    })
                continue

            if tool_calls:
                from .session_render import tool_call_card_from_args

                if get_setting(config, "tool_call_display") == "print":
                    console.print()
                append_status_history_items()

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

                tool_hooks = ToolExecutionHooks(
                    on_parallel_read=_on_parallel_read,
                    on_parallel_web_search=_on_parallel_web_search,
                    run_command=lambda args: _dispatch_run_command_with_ui(
                        args,
                        config,
                        history,
                        usage_path=usage_path,
                        session_id=session_context.session_id,
                        cancellation_token=cancellation_token,
                        retained_store=retained_store,
                        ui=ui,
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

                active_tool_call = tool_calls[0] if tool_calls else None

                def _execute_tools(_new_input, append_tool_result):
                    nonlocal active_tool_call, web_search_read_nudge_sent
                    active_tool_call = tool_calls[0] if tool_calls else None
                    result = execute_tool_calls(
                        tool_calls,
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
                    active_tool_call = None
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
            else:
                append_status_history_items()
                history.append({"role": "assistant", "content": reply_text, **metadata})
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
            try:
                pending.process.kill_tree()
            except Exception:
                pass
            if callable(getattr(pending, "unregister_cancel", None)):
                pending.unregister_cancel()
        if session_context is not None:
            _checkpoint_cancelled_turn()
        return AgentRunResult(cancelled=True, prompt=query)
    except (ProviderError, Exception) as e:
        label = "API error" if isinstance(e, ProviderError) else "Unexpected error"
        message = f"{label}: {e}"
        if ui is not None:
            _ui_call(ui, "show_error", message)
        else:
            style = "red"
            console.print(f"[{style}]{label}:[/{style}] {escape(str(e))}")
        _flush_error_state()
        return AgentRunResult(error=str(e))
    finally:
        sigint_cancel_scope.__exit__(None, None, None)
        _ui_call(ui, "unbind_cancel_token")
        _stop_live_displays()


