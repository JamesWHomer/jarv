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
from .config import DEFAULT_CONFIG
from .context_budget import trim_turn_input
from .safety import check_command
from .anthropic_http import DEFAULT_SUBAGENT_MAX_TOKENS
from .provider import (
    ProviderError,
    StreamDone,
    TextDelta,
    ToolCallDone,
    ReasoningDone,
    get_backend,
    responses_input_id,
    stream_response,
)
from .provider_catalog import configured_service_tier
from .read_tool import (
    READ_TOOL,
    dispatch_read_batch,
    dispatch_read_tool,
    retain_command_output,
)
from .retained_outputs import RetainedOutputStore
from .shell import (
    COMMAND_OUTPUT_UNSET,
    execute_command,
    resolve_command_output_window,
    truncate_model_output,
)
from .usage import estimate_context_breakdown, record_response_usage
from .web import WEB_SEARCH_TOOL, dispatch_web_tool


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
                "description": (
                    "Number of characters to return from the start of the output. "
                    "Defaults to half of max_tool_output_chars."
                ),
            },
            "tail_chars": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Number of characters to return from the end of the output. "
                    "Defaults to the remaining half of max_tool_output_chars."
                ),
            },
        },
        "required": ["command"],
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
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Unique handle for this child's artifact."},
                        "task": {"type": "string", "description": "Free-form instructions for the child."},
                        "sterile": {"type": "boolean", "description": "If true (default), child cannot spawn."},
                        "deps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Labels of prior artifacts to make visible to this child.",
                        },
                    },
                    "required": ["label", "task"],
                },
            }
        },
        "required": ["children"],
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
    },
}

class DepthExceeded(Exception):
    pass


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


def build_subagent_tools(sterile: bool) -> list[dict]:
    tools = [
        RUN_COMMAND_TOOL,
        WEB_SEARCH_TOOL,
        READ_TOOL,
        FINISH_TOOL,
    ]
    if not sterile:
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
    on_run_command: Callable[[str], str] | None = None,
    spawn_observer: "SpawnObserver | None" = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> str:
    """Execute a non-finish tool call and return the model-visible output string.

    `on_run_command` lets the root agent override run_command rendering with
    its rich UI. Subagents pass None and use the silent default.
    """
    if name == "run_command":
        cmd = args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            return "[tool argument error: command must be a non-empty string]"
        try:
            head_chars, tail_chars = resolve_command_output_window(
                args.get("head_chars", COMMAND_OUTPUT_UNSET),
                args.get("tail_chars", COMMAND_OUTPUT_UNSET),
                config.get(
                    "max_tool_output_chars",
                    DEFAULT_CONFIG["max_tool_output_chars"],
                ),
            )
        except ValueError as e:
            return f"[tool argument error: {e}]"
        safety_level = config.get("command_safety", "risky")
        audit = config.get("audit", True)
        auditor_history = [{"role": "user", "content": node.task}] if node.task else None
        allowed, denial = check_command(
            cmd,
            safety_level,
            audit=audit,
            config=config,
            history=auditor_history,
            usage_path=None if node.incognito else node.usage_path,
            session_id=node.session_id,
            cancellation_token=cancellation_token,
        )
        if not allowed:
            return denial
        if on_run_command is not None:
            output, _ = retain_command_output(
                on_run_command(cmd),
                head_chars,
                tail_chars,
                retained_store,
                int(
                    config.get(
                        "max_tool_output_chars",
                        DEFAULT_CONFIG["max_tool_output_chars"],
                    )
                ),
            )
            return output
        result = execute_command(
            cmd,
            config.get("command_timeout", 60),
            cancellation_token=cancellation_token,
        )
        output, _ = retain_command_output(
            result.full_model_output(),
            head_chars,
            tail_chars,
            retained_store,
            int(
                config.get(
                    "max_tool_output_chars",
                    DEFAULT_CONFIG["max_tool_output_chars"],
                )
            ),
        )
        return output

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
        children = args.get("children")
        if not isinstance(children, list) or not children:
            return "[tool argument error: children must be a non-empty list]"
        try:
            results = spawn_batch(
                node,
                children,
                store,
                client,
                config,
                observer=spawn_observer,
                cancellation_token=cancellation_token,
                retained_store=retained_store,
            )
        except DepthExceeded as e:
            return f"[error: {e}]"
        except ValueError as e:
            return f"[tool argument error: {e}]"
        return json.dumps(results)

    return f"[unknown tool: {name}]"


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

    tools = build_subagent_tools(node.sterile)
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

    while True:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        tool_calls: list = []
        reasoning_items: list = []
        context_breakdown = estimate_context_breakdown(
            config["model"],
            kwargs["instructions"],
            kwargs["tools"],
            kwargs["input"],
        )
        try:
            final_response = None
            for event in stream_response(
                client, config,
                kwargs["model"], kwargs["instructions"],
                kwargs["tools"], kwargs["input"],
                reasoning=kwargs.get("reasoning"),
                prompt_cache_key=f"jarv:{node.session_id}" if node.session_id else None,
                max_tokens=max_tokens,
                cancellation_token=cancellation_token,
            ):
                if isinstance(event, ToolCallDone):
                    tool_calls.append(event)
                elif isinstance(event, ReasoningDone):
                    reasoning_items.append(event)
                elif isinstance(event, StreamDone):
                    final_response = event.response
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
                    output_text="\n".join(f"{item.name} {item.arguments}" for item in tool_calls),
                )
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
        for ri in reasoning_items:
            item = {
                "type": "reasoning",
                "id": responses_input_id(str(ri.id), "rs"),
                "summary": [],
            }
            if ri.provider_content:
                item["provider_content"] = ri.provider_content
            new_input.append(item)

        def append_tool_result(item, output: str) -> None:
            function_call = {
                "type": "function_call",
                "id": responses_input_id(str(item.id), "fc"),
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            }
            if item.provider_content:
                function_call["provider_content"] = item.provider_content
            new_input.append(function_call)
            new_input.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": output,
            })

        item_index = 0
        while item_index < len(tool_calls):
            item = tool_calls[item_index]
            if item.name == "read":
                group_end = item_index
                while (
                    group_end < len(tool_calls)
                    and tool_calls[group_end].name == "read"
                ):
                    group_end += 1
                group = tool_calls[item_index:group_end]
                outputs = [""] * len(group)
                valid_args: list[dict] = []
                valid_indexes: list[int] = []
                for group_index, read_item in enumerate(group):
                    try:
                        read_args = json.loads(read_item.arguments or "{}")
                    except json.JSONDecodeError as e:
                        outputs[group_index] = (
                            f"[tool argument error: invalid JSON: {e}]"
                        )
                    else:
                        if not isinstance(read_args, dict):
                            outputs[group_index] = (
                                "[tool argument error: read arguments "
                                "must be an object]"
                            )
                        else:
                            valid_args.append(read_args)
                            valid_indexes.append(group_index)
                if valid_args:
                    batch_outputs = dispatch_read_batch(
                        valid_args,
                        visible_labels=node.visible_labels,
                        artifact_store=store,
                        retained_store=retained_store,
                        config=config,
                        cancellation_token=cancellation_token,
                    )
                    for group_index, output in zip(valid_indexes, batch_outputs):
                        outputs[group_index] = output
                for read_item, output in zip(group, outputs):
                    append_tool_result(read_item, output)
                item_index = group_end
                continue

            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"[tool argument error: invalid JSON: {e}]"
            else:
                if item.name == "finish":
                    longform = args.get("longform")
                    tldr = args.get("tldr")
                    if not isinstance(longform, str) or not isinstance(tldr, str):
                        output = "[finish requires string longform and tldr]"
                    else:
                        return longform, tldr
                else:
                    output = dispatch_tool(
                        item.name, args, node, store, client, config,
                        spawn_observer=spawn_observer,
                        cancellation_token=cancellation_token,
                        retained_store=retained_store,
                    )

            if item.name != "run_command":
                output = truncate_model_output(
                    output,
                    config.get(
                        "max_tool_output_chars",
                        DEFAULT_CONFIG["max_tool_output_chars"],
                    ),
                )
            append_tool_result(item, output)
            item_index += 1

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
        label = spec.get("label")
        task = spec.get("task")
        if not isinstance(label, str) or not label:
            raise ValueError("each child needs a non-empty 'label'")
        if label in seen_labels:
            raise ValueError(f"duplicate child label '{label}'")
        seen_labels.add(label)
        if not isinstance(task, str) or not task:
            raise ValueError(f"child '{label}' needs a non-empty 'task'")
        sterile = bool(spec.get("sterile", True))
        raw_deps = spec.get("deps") or []
        if not isinstance(raw_deps, list):
            raise ValueError(f"child '{label}' deps must be a list")
        valid_deps = {d for d in raw_deps if isinstance(d, str) and d in parent.visible_labels}
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
