"""Recursive subagent orchestration runtime.

Every agent is the same loop. Root is depth 0. Each `spawn` increments depth
for its children. Children run in parallel; the parent blocks until all
children terminate. Subagents emit `(longform, tldr)` via a terminal `finish`
tool; only the artifact persists, transcripts are discarded.
"""

import concurrent.futures
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .artifacts import ArtifactStore
from .cancellation import CancellationToken, TurnCancelled
from .config import DEFAULT_CONFIG, TOOL_NAMES
from .context_budget import trim_turn_input
from .safety import check_command
from .anthropic_http import DEFAULT_SUBAGENT_MAX_TOKENS
from .provider import (
    ProviderError,
    RetryableStreamError,
    get_backend,
    stream_response,
)
from .provider_catalog import configured_service_tier
from .read_tool import (
    dispatch_read_tool,
    read_tool_for_config,
    retain_command_output,
)
from .retained_outputs import RetainedOutputStore
from .shell import (
    COMMAND_OUTPUT_UNSET,
    MAX_COMMAND_OUTPUT_WINDOW_CHARS,
    execute_command,
    resolve_command_output_window,
    truncate_model_output,
)
from .tool_outputs import ToolOutput
from .turn_loop import collect_stream_response
from .turn_records import (
    append_reasoning_input_items,
    append_tool_result_input_items,
    stream_usage_output_text,
)
from .usage import estimate_context_breakdown, record_response_usage
from .web import WEB_SEARCH_TOOL, dispatch_web_tool


MAX_SPAWN_CHILDREN = 16
MAX_SPAWN_DEPS = 32
MAX_SPAWN_LABEL_CHARS = 80
MAX_SPAWN_TASK_CHARS = 50_000

RUN_COMMAND_TOOL = {
    "type": "function",
    "name": "run_command",
    "description": "Run a shell command and return its output.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute"},
            "head_chars": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_COMMAND_OUTPUT_WINDOW_CHARS,
                "description": (
                    "Number of characters to return from the start of the output. "
                    "Defaults to half of max_tool_output_chars."
                ),
            },
            "tail_chars": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_COMMAND_OUTPUT_WINDOW_CHARS,
                "description": (
                    "Number of characters to return from the end of the output. "
                    "Defaults to the remaining half of max_tool_output_chars."
                ),
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
}

SPAWN_TOOL = {
    "type": "function",
    "name": "spawn",
    "description": (
        "Fan out work to N parallel subagents. Blocks until all children finish. "
        "Each child gets its `task`, plus the (label, tldr) of every artifact named in `deps`. "
        "Children can call read on those labels for full content. "
        "sterile=true (default) means the child cannot itself spawn. "
        "Returns one entry per child: {label, status: 'done'|'failed', tldr?, reason?}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "children": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_SPAWN_CHILDREN,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_SPAWN_LABEL_CHARS,
                            "description": "Unique handle for this child's artifact.",
                        },
                        "task": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_SPAWN_TASK_CHARS,
                            "description": "Free-form instructions for the child.",
                        },
                        "sterile": {"type": "boolean", "description": "If true (default), child cannot spawn."},
                        "deps": {
                            "type": "array",
                            "maxItems": MAX_SPAWN_DEPS,
                            "items": {"type": "string"},
                            "description": "Labels of prior artifacts to make visible to this child.",
                        },
                    },
                    "required": ["label", "task"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["children"],
        "additionalProperties": False,
    },
}

FINISH_TOOL = {
    "type": "function",
    "name": "finish",
    "description": (
        "YOU MUST CALL THIS. It is the only way to return output — any text you write outside a tool call is invisible and discarded. "
        "Call exactly once when your task is complete. No exceptions, even for trivial tasks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "longform": {"type": "string", "description": "Full report or result. Parent reads this on demand via read."},
            "tldr": {"type": "string", "description": "1-2 sentence summary inlined into the parent's next turn."},
        },
        "required": ["longform", "tldr"],
        "additionalProperties": False,
    },
}

ASK_USER_TOOL = {
    "type": "function",
    "name": "ask_user",
    "description": (
        "Ask the user a question and wait for their response. "
        "Use when you need clarification, a choice between options, or confirmation before proceeding. "
        "The conversation pauses until the user replies."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The Markdown-formatted question to present to the user.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

class DepthExceeded(Exception):
    pass


PARALLEL_SAFE_TOOL_NAMES = {"read", "web_search"}
WEB_SEARCH_READ_NUDGE = (
    "[Jarv tool note: To inspect result pages, call `read(input=<URL>)`; "
    "read multiple independent URLs in the same response.]"
)


@dataclass
class AgentNode:
    label: str
    depth: int
    parent_label: str | None
    task: str
    sterile: bool
    visible_labels: set[str] = field(default_factory=set)
    usage_path: Path | None = None
    session_id: str | None = None
    incognito: bool = False


def tool_enabled(config: dict, name: str) -> bool:
    """Return whether a user-facing tool is enabled in config."""
    disabled = config.get("disabled_tools", [])
    return name in TOOL_NAMES and (
        not isinstance(disabled, list) or name not in disabled
    )


def filter_enabled_tools(tools: list[dict], config: dict) -> list[dict]:
    """Filter user-facing tool definitions using the current config."""
    return [
        tool
        for tool in tools
        if tool_enabled(config, str(tool.get("name", "")))
    ]


@dataclass(frozen=True)
class ToolBatchResult:
    args: dict | None
    output: ToolOutput


@dataclass
class ToolExecutionHooks:
    """Optional root-agent UI hooks; subagents leave these unset and use dispatch_tool."""

    on_parallel_read: Callable[[object, dict], None] | None = None
    on_parallel_web_search: Callable[[object, dict], None] | None = None
    run_command: Callable[[dict], str] | None = None
    run_spawn: Callable[[dict], str] | None = None
    run_ask_user: Callable[[dict], str] | None = None
    on_tool_error: Callable[[str], None] | None = None


@dataclass
class ToolExecutionResult:
    finished: tuple[str, str] | None = None
    web_search_read_nudge_sent: bool = False


def tool_call_is_parallel_safe(name: str) -> bool:
    return name in PARALLEL_SAFE_TOOL_NAMES


def _optional_arg(args: dict, name: str, default):
    value = args.get(name, default)
    return default if value is None else value


@dataclass(frozen=True)
class RunCommandPrepared:
    cmd: str
    head_chars: int
    tail_chars: int
    max_tool_output_chars: int


def prepare_run_command(args: dict, config: dict) -> RunCommandPrepared | str:
    """Validate run_command args and resolve the model output window."""
    cmd = args.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return "[tool argument error: command must be a non-empty string]"
    try:
        head_chars, tail_chars = resolve_command_output_window(
            _optional_arg(args, "head_chars", COMMAND_OUTPUT_UNSET),
            _optional_arg(args, "tail_chars", COMMAND_OUTPUT_UNSET),
            config.get(
                "max_tool_output_chars",
                DEFAULT_CONFIG["max_tool_output_chars"],
            ),
        )
    except ValueError as e:
        return f"[tool argument error: {e}]"
    return RunCommandPrepared(
        cmd=cmd,
        head_chars=head_chars,
        tail_chars=tail_chars,
        max_tool_output_chars=int(
            config.get(
                "max_tool_output_chars",
                DEFAULT_CONFIG["max_tool_output_chars"],
            )
        ),
    )


def check_run_command(
    prepared: RunCommandPrepared,
    config: dict,
    *,
    safety_history: list | None = None,
    usage_path: Path | None = None,
    session_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    incognito: bool = False,
) -> tuple[bool, str]:
    """Run command-safety checks. Returns (allowed, denial_message)."""
    safety_level = config.get("command_safety", "risky")
    audit = config.get("audit", True)
    return check_command(
        prepared.cmd,
        safety_level,
        audit=audit,
        config=config,
        history=safety_history,
        usage_path=None if incognito else usage_path,
        session_id=session_id,
        cancellation_token=cancellation_token,
    )


def execute_run_command(
    prepared: RunCommandPrepared,
    config: dict,
    *,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> tuple[str, object, str | None]:
    """Execute a prepared shell command and return model output plus shell metadata."""
    result = execute_command(
        prepared.cmd,
        config.get("command_timeout", 60),
        cancellation_token=cancellation_token,
    )
    output, output_id = format_run_command_output(
        result,
        prepared,
        retained_store,
    )
    return output, result, output_id


def format_run_command_output(
    result,
    prepared: RunCommandPrepared,
    retained_store: RetainedOutputStore | None,
) -> tuple[str, str | None]:
    """Retain and window shell output for the model."""
    return retain_command_output(
        result.full_model_output(),
        prepared.head_chars,
        prepared.tail_chars,
        retained_store,
        prepared.max_tool_output_chars,
    )


def run_command_tool_output(
    args: dict,
    config: dict,
    *,
    safety_history: list | None = None,
    usage_path: Path | None = None,
    session_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
    incognito: bool = False,
) -> str:
    """Validate, safety-check, execute, and retain output for run_command."""
    prepared = prepare_run_command(args, config)
    if isinstance(prepared, str):
        return prepared
    allowed, denial = check_run_command(
        prepared,
        config,
        safety_history=safety_history,
        usage_path=usage_path,
        session_id=session_id,
        cancellation_token=cancellation_token,
        incognito=incognito,
    )
    if not allowed:
        return denial
    output, _, _ = execute_run_command(
        prepared,
        config,
        cancellation_token=cancellation_token,
        retained_store=retained_store,
    )
    return output


def parse_spawn_children(args: dict) -> list | str:
    """Return child specs or a model-visible error string."""
    children = args.get("children")
    if not isinstance(children, list) or not children:
        return "[tool argument error: children must be a non-empty list]"
    if len(children) > MAX_SPAWN_CHILDREN:
        return (
            "[tool argument error: children must contain at most "
            f"{MAX_SPAWN_CHILDREN} entries]"
        )
    return children


def spawn_tool_output(
    node: AgentNode,
    children: list,
    store: ArtifactStore,
    client,
    config: dict,
    *,
    spawn_observer: "SpawnObserver | None" = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> str:
    """Fan out spawn children and return the model-visible JSON status report."""
    try:
        results = spawn_batch(
            node,
            children,
            store,
            client,
            config,
            observer=spawn_observer,
            usage_path=node.usage_path,
            session_id=node.session_id,
            cancellation_token=cancellation_token,
            retained_store=retained_store,
        )
    except DepthExceeded as e:
        return f"[error: {e}]"
    except ValueError as e:
        return f"[tool argument error: {e}]"
    return json.dumps(results)


def history_has_web_search_read_nudge(items: list[dict]) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call_output":
            continue
        output = item.get("output")
        if isinstance(output, str) and WEB_SEARCH_READ_NUDGE in output:
            return True
    return False


def append_web_search_read_nudge(output: str) -> str:
    if not output:
        return WEB_SEARCH_READ_NUDGE
    return output.rstrip() + "\n\n" + WEB_SEARCH_READ_NUDGE


def _parse_tool_args(arguments: str | None) -> tuple[dict | None, str | None]:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return None, f"[tool argument error: invalid JSON: {e}]"
    if not isinstance(args, dict):
        return None, "[tool argument error: arguments must be an object]"
    return args, None


def dispatch_parallel_safe_tool_batch(
    tool_calls: list,
    *,
    node: AgentNode,
    store: ArtifactStore,
    client,
    config: dict,
    spawn_observer: "SpawnObserver | None" = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> list[ToolBatchResult]:
    results = [ToolBatchResult(None, "") for _ in tool_calls]
    valid: list[tuple[int, object, dict]] = []

    for index, item in enumerate(tool_calls):
        if not tool_call_is_parallel_safe(item.name):
            results[index] = ToolBatchResult(
                None,
                f"[tool not parallel-safe: {item.name}]",
            )
            continue
        if item.name in TOOL_NAMES and not tool_enabled(config, item.name):
            results[index] = ToolBatchResult(
                None,
                f"[tool disabled: {item.name}]",
            )
            continue
        args, error = _parse_tool_args(item.arguments)
        if error is not None:
            results[index] = ToolBatchResult(None, error)
            continue
        assert args is not None
        valid.append((index, item, args))

    if not valid:
        return results

    def run_one(item, args: dict) -> ToolOutput:
        return dispatch_tool(
            item.name,
            args,
            node,
            store,
            client,
            config,
            spawn_observer=spawn_observer,
            cancellation_token=cancellation_token,
            retained_store=retained_store,
        )

    if len(valid) == 1:
        index, item, args = valid[0]
        results[index] = ToolBatchResult(args, run_one(item, args))
        return results

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(valid))
    )
    futures = {
        executor.submit(run_one, item, args): (index, args)
        for index, item, args in valid
    }
    try:
        for future in concurrent.futures.as_completed(futures):
            index, args = futures[future]
            results[index] = ToolBatchResult(args, future.result())
    except (KeyboardInterrupt, TurnCancelled):
        if cancellation_token is not None:
            cancellation_token.cancel()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except Exception:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def build_subagent_tools(sterile: bool, config: dict | None = None) -> list[dict]:
    config = config or DEFAULT_CONFIG
    tools = filter_enabled_tools([
        RUN_COMMAND_TOOL,
        WEB_SEARCH_TOOL,
        read_tool_for_config(config),
    ], config)
    tools.append(FINISH_TOOL)
    if not sterile and tool_enabled(config, "spawn"):
        tools.append(SPAWN_TOOL)
    return tools


def _format_deps_block(node: AgentNode, store: ArtifactStore) -> str:
    if not node.visible_labels:
        return ""
    lines = []
    for lbl in sorted(node.visible_labels):
        art = store.get(lbl)
        if art is not None:
            lines.append(f"- {lbl}: {art.tldr}")
    if not lines:
        return ""
    return (
        "\n\nVisible artifacts (call read(input=label) for the full longform):\n"
        + "\n".join(lines)
    )


def dispatch_tool(
    name: str,
    args: dict,
    node: AgentNode,
    store: ArtifactStore,
    client,
    config: dict,
    spawn_observer: "SpawnObserver | None" = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> ToolOutput:
    """Execute a non-finish tool call and return the model-visible output string."""
    if name in TOOL_NAMES and not tool_enabled(config, name):
        return f"[tool disabled: {name}]"

    if name == "run_command":
        return run_command_tool_output(
            args,
            config,
            safety_history=(
                [{"role": "user", "content": node.task}] if node.task else None
            ),
            usage_path=node.usage_path,
            session_id=node.session_id,
            cancellation_token=cancellation_token,
            retained_store=retained_store,
            incognito=node.incognito,
        )

    if name == "read":
        return dispatch_read_tool(
            args,
            visible_labels=node.visible_labels,
            artifact_store=store,
            retained_store=retained_store or RetainedOutputStore(),
            config=config,
            cancellation_token=cancellation_token,
        )

    if name == "web_search":
        return dispatch_web_tool(
            name,
            args,
            config,
            cancellation_token=cancellation_token,
        )

    if name == "spawn":
        children = parse_spawn_children(args)
        if isinstance(children, str):
            return children
        return spawn_tool_output(
            node,
            children,
            store,
            client,
            config,
            spawn_observer=spawn_observer,
            cancellation_token=cancellation_token,
            retained_store=retained_store,
        )

    return f"[unknown tool: {name}]"


def _max_tool_output_chars(config: dict) -> int:
    return int(
        config.get(
            "max_tool_output_chars",
            DEFAULT_CONFIG["max_tool_output_chars"],
        )
    )


def _maybe_truncate_tool_output(name: str, output: ToolOutput, config: dict) -> ToolOutput:
    if name in {"run_command", "read"}:
        return output
    return truncate_model_output(output, _max_tool_output_chars(config))


def execute_tool_calls(
    tool_calls: list,
    *,
    node: AgentNode,
    store: ArtifactStore,
    client,
    config: dict,
    append_tool_result: Callable[[object, ToolOutput], None],
    hooks: ToolExecutionHooks | None = None,
    spawn_observer: "SpawnObserver | None" = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
    web_search_read_nudge_sent: bool = False,
) -> ToolExecutionResult:
    """Run a model tool-call batch, grouping parallel-safe tools when possible."""
    hooks = hooks or ToolExecutionHooks()
    result = ToolExecutionResult(web_search_read_nudge_sent=web_search_read_nudge_sent)

    item_index = 0
    while item_index < len(tool_calls):
        item = tool_calls[item_index]
        if tool_call_is_parallel_safe(item.name):
            group_end = item_index
            while (
                group_end < len(tool_calls)
                and tool_call_is_parallel_safe(tool_calls[group_end].name)
            ):
                group_end += 1
            group = tool_calls[item_index:group_end]
            batch_results = dispatch_parallel_safe_tool_batch(
                group,
                node=node,
                store=store,
                client=client,
                config=config,
                spawn_observer=spawn_observer,
                cancellation_token=cancellation_token,
                retained_store=retained_store,
            )
            for safe_item, batch_result in zip(group, batch_results):
                output = batch_result.output
                if safe_item.name == "read" and batch_result.args is not None:
                    if hooks.on_parallel_read is not None:
                        hooks.on_parallel_read(safe_item, batch_result.args)
                elif safe_item.name == "web_search" and batch_result.args is not None:
                    if hooks.on_parallel_web_search is not None:
                        hooks.on_parallel_web_search(safe_item, batch_result.args)
                    if not result.web_search_read_nudge_sent:
                        output = append_web_search_read_nudge(output)
                        result.web_search_read_nudge_sent = True
                output = _maybe_truncate_tool_output(safe_item.name, output, config)
                append_tool_result(safe_item, output)
            item_index = group_end
            continue

        if item.name in TOOL_NAMES and not tool_enabled(config, item.name):
            append_tool_result(item, f"[tool disabled: {item.name}]")
            item_index += 1
            continue

        try:
            args = json.loads(item.arguments or "{}")
        except json.JSONDecodeError as e:
            output = f"[tool argument error: invalid JSON: {e}]"
            if hooks.on_tool_error is not None:
                hooks.on_tool_error(output)
            append_tool_result(item, output)
            item_index += 1
            continue

        if item.name == "finish":
            longform = args.get("longform")
            tldr = args.get("tldr")
            if not isinstance(longform, str) or not isinstance(tldr, str):
                append_tool_result(item, "[finish requires string longform and tldr]")
            else:
                result.finished = (longform, tldr)
                return result
            item_index += 1
            continue

        if item.name == "run_command" and hooks.run_command is not None:
            output = hooks.run_command(args)
        elif item.name == "spawn" and hooks.run_spawn is not None:
            output = hooks.run_spawn(args)
        elif item.name == "ask_user" and hooks.run_ask_user is not None:
            output = hooks.run_ask_user(args)
        else:
            output = dispatch_tool(
                item.name,
                args,
                node,
                store,
                client,
                config,
                spawn_observer=spawn_observer,
                cancellation_token=cancellation_token,
                retained_store=retained_store,
            )

        if hooks.on_tool_error is not None and isinstance(output, str):
            if output.startswith("[unknown tool:") or output.startswith("[tool argument error:"):
                hooks.on_tool_error(output)

        output = _maybe_truncate_tool_output(item.name, output, config)
        append_tool_result(item, output)
        item_index += 1

    return result


_FINISH_NUDGE = (
    "You must call the finish tool to complete your task. "
    "Plain text outside a tool call is discarded and will fail this run. "
    "Call finish(longform, tldr) now."
)

_FINISH_TRUNCATION_NUDGE = (
    "Your finish() call was truncated by the output token limit. "
    "Call finish(longform, tldr) again with a shorter longform."
)


def _subagent_max_tokens(config: dict) -> int | None:
    if get_backend(config) != "anthropic":
        return None
    return int(config.get("anthropic_subagent_max_tokens", DEFAULT_SUBAGENT_MAX_TOKENS))


def run_subagent_loop(
    node: AgentNode,
    store: ArtifactStore,
    client,
    config: dict,
    spawn_observer: "SpawnObserver | None" = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> tuple[str | None, str]:
    """Run a single subagent to completion.

    Returns (longform, tldr) on success, (None, reason) on failure.
    """
    retained_store = retained_store or RetainedOutputStore()
    instructions = (
        "You are a subagent in a recursive orchestration system. "
        "Complete your task, then call finish(longform, tldr) to terminate — this is mandatory. "
        "Any text you write outside a tool call is invisible to the parent and will be discarded. "
        "finish() is the only way your output is ever seen. You must call it even for the simplest task."
    ) + _format_deps_block(node, store)

    tools = build_subagent_tools(node.sterile, config)
    input_items: list[dict] = [{"role": "user", "content": node.task}]

    kwargs = dict(
        model=config["model"],
        instructions=instructions,
        tools=tools,
        input=input_items,
    )
    effort = config.get("reasoning_effort")
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    max_tokens = _subagent_max_tokens(config)
    finish_nudge_sent = False
    finish_truncation_nudge_sent = False
    web_search_read_nudge_sent = False

    while True:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        context_breakdown = estimate_context_breakdown(
            config["model"],
            kwargs["instructions"],
            kwargs["tools"],
            kwargs["input"],
        )

        def make_stream():
            return stream_response(
                client, config,
                kwargs["model"], kwargs["instructions"],
                kwargs["tools"], kwargs["input"],
                reasoning=kwargs.get("reasoning"),
                prompt_cache_key=f"jarv:{node.session_id}" if node.session_id else None,
                max_tokens=max_tokens,
                cancellation_token=cancellation_token,
            )

        try:
            stream_result = collect_stream_response(make_stream)
            tool_calls = stream_result.tool_calls
            reasoning_items = stream_result.reasoning_items
            final_response = stream_result.final_response
            if not node.incognito:
                record_response_usage(
                    node.usage_path,
                    node.session_id,
                    config["model"],
                    final_response,
                    "subagent",
                    provider=str(config.get("provider") or "openai"),
                    requested_service_tier=configured_service_tier(config),
                    context_breakdown=context_breakdown,
                    output_text=stream_usage_output_text("", tool_calls),
                )
        except RetryableStreamError as e:
            return None, f"provider error: {e}"
        except ProviderError as e:
            return None, f"provider error: {e}"
        except TurnCancelled:
            raise
        except Exception as e:
            return None, f"stream error: {e}"

        if not tool_calls:
            if not finish_nudge_sent:
                finish_nudge_sent = True
                kwargs["input"] = kwargs["input"] + [{"role": "user", "content": _FINISH_NUDGE}]
                continue
            return None, "subagent terminated without calling finish"

        stop_reason = (
            final_response.get("stop_reason")
            if isinstance(final_response, dict) else None
        )
        truncated_finish = False
        for item in tool_calls:
            if item.name != "finish":
                continue
            try:
                json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                if stop_reason == "max_tokens" and not finish_truncation_nudge_sent:
                    finish_truncation_nudge_sent = True
                    truncated_finish = True
                    break

        if truncated_finish:
            kwargs["input"] = kwargs["input"] + [{"role": "user", "content": _FINISH_TRUNCATION_NUDGE}]
            continue

        new_input: list[dict] = []
        append_reasoning_input_items(new_input, reasoning_items)

        def append_tool_result(item, output: ToolOutput) -> None:
            append_tool_result_input_items(new_input, item, output)

        exec_result = execute_tool_calls(
            tool_calls,
            node=node,
            store=store,
            client=client,
            config=config,
            append_tool_result=append_tool_result,
            spawn_observer=spawn_observer,
            cancellation_token=cancellation_token,
            retained_store=retained_store,
            web_search_read_nudge_sent=web_search_read_nudge_sent,
        )
        web_search_read_nudge_sent = exec_result.web_search_read_nudge_sent
        if exec_result.finished is not None:
            return exec_result.finished

        kwargs["input"] = trim_turn_input(
            kwargs["input"] + new_input,
            model=config["model"],
            config=config,
            instructions=kwargs["instructions"],
            tools=kwargs["tools"],
        )


class SpawnObserver:
    """Hook surface for the UI to observe nested spawn activity.

    All methods are called from worker threads; implementations must be
    thread-safe.
    """

    def on_spawn_start(self, parent_label: str, child_labels: list[str]) -> None:
        pass

    def on_child_done(self, parent_label: str, label: str, result: dict) -> None:
        pass


def spawn_batch(
    parent: AgentNode,
    child_specs: list[dict],
    store: ArtifactStore,
    client,
    config: dict,
    observer: "SpawnObserver | None" = None,
    usage_path: Path | None = None,
    session_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> list[dict]:
    """Spawn N children in parallel, block until all finish, return status reports."""
    if len(child_specs) > MAX_SPAWN_CHILDREN:
        raise ValueError(
            f"children must contain at most {MAX_SPAWN_CHILDREN} entries"
        )
    new_depth = parent.depth + 1
    max_depth = int(config.get("max_subagent_depth", 4))
    if new_depth > max_depth:
        raise DepthExceeded(
            f"depth cap {max_depth} reached (this spawn would create depth {new_depth} children)"
        )

    nodes: list[AgentNode] = []
    seen_labels: set[str] = set()
    child_usage_path = usage_path if usage_path is not None else parent.usage_path
    child_session_id = session_id if session_id is not None else parent.session_id
    for spec in child_specs:
        if not isinstance(spec, dict):
            raise ValueError(f"child spec must be an object, got {type(spec).__name__}")
        extra_keys = set(spec) - {"label", "task", "sterile", "deps"}
        if extra_keys:
            keys = ", ".join(sorted(extra_keys))
            raise ValueError(f"child spec contains unknown fields: {keys}")
        label = spec.get("label")
        task = spec.get("task")
        if not isinstance(label, str) or not label:
            raise ValueError("each child needs a non-empty 'label'")
        if len(label) > MAX_SPAWN_LABEL_CHARS:
            raise ValueError(
                f"child label must be at most {MAX_SPAWN_LABEL_CHARS} characters"
            )
        if label in seen_labels:
            raise ValueError(f"duplicate child label '{label}'")
        seen_labels.add(label)
        if not isinstance(task, str) or not task:
            raise ValueError(f"child '{label}' needs a non-empty 'task'")
        if len(task) > MAX_SPAWN_TASK_CHARS:
            raise ValueError(
                f"child '{label}' task must be at most {MAX_SPAWN_TASK_CHARS} characters"
            )
        sterile_value = spec.get("sterile", True)
        if sterile_value is None:
            sterile_value = True
        if not isinstance(sterile_value, bool):
            raise ValueError(f"child '{label}' sterile must be a boolean")
        sterile = sterile_value
        raw_deps = spec.get("deps", [])
        if raw_deps is None:
            raw_deps = []
        if not isinstance(raw_deps, list):
            raise ValueError(f"child '{label}' deps must be a list")
        if len(raw_deps) > MAX_SPAWN_DEPS:
            raise ValueError(
                f"child '{label}' deps must contain at most {MAX_SPAWN_DEPS} entries"
            )
        invalid_deps = [
            d for d in raw_deps
            if not isinstance(d, str) or d not in parent.visible_labels
        ]
        if invalid_deps:
            invalid = ", ".join(repr(d) for d in invalid_deps[:5])
            suffix = "" if len(invalid_deps) <= 5 else ", ..."
            raise ValueError(f"child '{label}' has invalid deps: {invalid}{suffix}")
        valid_deps = set(raw_deps)
        nodes.append(AgentNode(
            label=label,
            depth=new_depth,
            parent_label=parent.label,
            task=task,
            sterile=sterile,
            visible_labels=valid_deps,
            usage_path=child_usage_path,
            session_id=child_session_id,
            incognito=parent.incognito,
        ))

    if observer is not None:
        observer.on_spawn_start(parent.label, [n.label for n in nodes])

    pool_size = max(1, int(config.get("subagent_thread_pool_max_workers", 8)))
    raw_results: dict[str, dict] = {}
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)
    future_to_node = {}
    try:
        future_to_node = {
            ex.submit(
                run_subagent_loop,
                n,
                store,
                client,
                config,
                spawn_observer=observer,
                cancellation_token=cancellation_token,
                retained_store=retained_store,
            ): n
            for n in nodes
        }
        pending = set(future_to_node)
        while pending:
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            done, pending = concurrent.futures.wait(
                pending,
                timeout=0.05,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                n = future_to_node[fut]
                try:
                    longform, tldr_or_reason = fut.result()
                except TurnCancelled:
                    raise
                except Exception as e:
                    result = {"label": n.label, "status": "failed", "reason": f"unhandled exception: {e}"}
                else:
                    if longform is not None:
                        store.put(n.label, longform, tldr_or_reason, n.label)
                        parent.visible_labels.add(n.label)
                        result = {"label": n.label, "status": "done", "tldr": tldr_or_reason}
                    else:
                        result = {"label": n.label, "status": "failed", "reason": tldr_or_reason}
                raw_results[n.label] = result
                if observer is not None:
                    observer.on_child_done(parent.label, n.label, result)
    except (KeyboardInterrupt, TurnCancelled):
        if cancellation_token is not None:
            cancellation_token.cancel()
        raise
    finally:
        if cancellation_token is not None and cancellation_token.cancelled:
            for future in future_to_node:
                future.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
        else:
            ex.shutdown(wait=True)

    return [raw_results[spec["label"]] for spec in child_specs if spec.get("label") in raw_results]
