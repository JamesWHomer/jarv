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
from .provider import (
    ProviderError,
    StreamDone,
    TextDelta,
    ToolCallDone,
    ReasoningDone,
    stream_response,
)
from .shell import execute_command
from .usage import record_response_usage


RUN_COMMAND_TOOL = {
    "type": "function",
    "name": "run_command",
    "description": "Run a shell command and return its output.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute"}
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
        "Children can call read_artifact on those labels for full content. "
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
            "longform": {"type": "string", "description": "Full report or result. Parent reads this on demand via read_artifact."},
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
            "question": {"type": "string", "description": "The question to present to the user."},
        },
        "required": ["question"],
    },
}

READ_ARTIFACT_TOOL = {
    "type": "function",
    "name": "read_artifact",
    "description": "Fetch the longform of an artifact whose label is currently visible to you.",
    "parameters": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
        },
        "required": ["label"],
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


def build_subagent_tools(sterile: bool) -> list[dict]:
    tools = [RUN_COMMAND_TOOL, READ_ARTIFACT_TOOL, FINISH_TOOL]
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
        "\n\nVisible artifacts (call read_artifact(label) for the full longform):\n"
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
) -> str:
    """Execute a non-finish tool call and return the model-visible output string.

    `on_run_command` lets the root agent override run_command rendering with
    its rich UI. Subagents pass None and use the silent default.
    """
    if name == "run_command":
        cmd = args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            return "[tool argument error: command must be a non-empty string]"
        if on_run_command is not None:
            return on_run_command(cmd)
        result = execute_command(cmd, config.get("command_timeout", 60))
        return result.to_model_output()

    if name == "read_artifact":
        label = args.get("label")
        if not isinstance(label, str) or not label:
            return "[tool argument error: label required]"
        if label not in node.visible_labels:
            return f"[error: artifact '{label}' is not visible to this agent]"
        art = store.get(label)
        if art is None:
            return f"[error: artifact '{label}' not found in store]"
        return art.longform

    if name == "spawn":
        children = args.get("children")
        if not isinstance(children, list) or not children:
            return "[tool argument error: children must be a non-empty list]"
        try:
            results = spawn_batch(node, children, store, client, config, observer=spawn_observer)
        except DepthExceeded as e:
            return f"[error: {e}]"
        return json.dumps(results)

    return f"[unknown tool: {name}]"


def run_subagent_loop(
    node: AgentNode,
    store: ArtifactStore,
    client,
    config: dict,
    spawn_observer: "SpawnObserver | None" = None,
) -> tuple[str | None, str]:
    """Run a single subagent to completion.

    Returns (longform, tldr) on success, (None, reason) on failure.
    """
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

    while True:
        tool_calls: list = []
        reasoning_items: list = []
        try:
            final_response = None
            for event in stream_response(
                client, config,
                kwargs["model"], kwargs["instructions"],
                kwargs["tools"], kwargs["input"],
                reasoning=kwargs.get("reasoning"),
            ):
                if isinstance(event, ToolCallDone):
                    tool_calls.append(event)
                elif isinstance(event, ReasoningDone):
                    reasoning_items.append(event)
                elif isinstance(event, StreamDone):
                    final_response = event.response
            record_response_usage(
                node.usage_path,
                node.session_id,
                config["model"],
                final_response,
                "subagent",
            )
        except ProviderError as e:
            return None, f"provider error: {e}"
        except Exception as e:
            return None, f"stream error: {e}"

        if not tool_calls:
            return None, "subagent terminated without calling finish"

        new_input: list[dict] = []
        for ri in reasoning_items:
            new_input.append({"type": "reasoning", "id": ri.id, "summary": []})

        for item in tool_calls:
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
                    )

            new_input.append({
                "type": "function_call",
                "id": item.id,
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            })
            new_input.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": output,
            })

        kwargs["input"] = kwargs["input"] + new_input


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
) -> list[dict]:
    """Spawn N children in parallel, block until all finish, return status reports."""
    new_depth = parent.depth + 1
    max_depth = int(config.get("max_subagent_depth", 4))
    if new_depth > max_depth:
        raise DepthExceeded(
            f"depth cap {max_depth} reached (this spawn would create depth {new_depth} children)"
        )

    nodes: list[AgentNode] = []
    child_usage_path = usage_path if usage_path is not None else parent.usage_path
    child_session_id = session_id if session_id is not None else parent.session_id
    for spec in child_specs:
        if not isinstance(spec, dict):
            raise ValueError(f"child spec must be an object, got {type(spec).__name__}")
        label = spec.get("label")
        task = spec.get("task")
        if not isinstance(label, str) or not label:
            raise ValueError("each child needs a non-empty 'label'")
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
        ))

    if observer is not None:
        observer.on_spawn_start(parent.label, [n.label for n in nodes])

    pool_size = max(1, int(config.get("subagent_thread_pool_max_workers", 8)))
    raw_results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as ex:
        future_to_node = {
            ex.submit(run_subagent_loop, n, store, client, config, observer): n
            for n in nodes
        }
        for fut in concurrent.futures.as_completed(future_to_node):
            n = future_to_node[fut]
            try:
                longform, tldr_or_reason = fut.result()
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

    return [raw_results[spec["label"]] for spec in child_specs if spec.get("label") in raw_results]
