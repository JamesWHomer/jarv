from __future__ import annotations

import base64
import re
from dataclasses import asdict, dataclass
from typing import Any, TypeAlias


ToolOutput: TypeAlias = str | list[dict[str, Any]]


@dataclass(frozen=True)
class ToolOutcome:
    """Execution metadata, independent of the model-visible output."""

    status: str
    exit_code: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> ToolOutcome | None:
        if not isinstance(value, dict):
            return None
        status = value.get("status")
        if status not in (
            "success", "failed", "denied", "timed_out", "cancelled",
            "running", "skipped", "unknown",
        ):
            return None
        exit_code = value.get("exit_code")
        if exit_code is not None and type(exit_code) is not int:
            return None
        return cls(status, exit_code)


class _ToolText(str):
    def __new__(cls, output: str, outcome: ToolOutcome):
        result = super().__new__(cls, output)
        result.outcome = outcome
        return result

    def __getnewargs__(self):
        return str(self), self.outcome


class _ToolBlocks(list):
    def __init__(self, output: list, outcome: ToolOutcome):
        super().__init__(output)
        self.outcome = outcome


def with_tool_outcome(output: ToolOutput, outcome: ToolOutcome | str) -> ToolOutput:
    """Carry an outcome without changing existing string/block tool interfaces.

    History builders persist ``outcome`` separately; JSON/provider payloads
    remain ordinary strings or content blocks. Text transformations must use
    ``preserve_tool_outcome`` so windowing cannot discard execution metadata.
    """
    if isinstance(outcome, str):
        outcome = ToolOutcome(outcome)
    if isinstance(output, list):
        return _ToolBlocks(output, outcome)
    return _ToolText(output, outcome)


def tool_outcome(output: ToolOutput) -> ToolOutcome | None:
    return getattr(output, "outcome", None)


def preserve_tool_outcome(output: ToolOutput, source: ToolOutput) -> ToolOutput:
    outcome = tool_outcome(source)
    return with_tool_outcome(output, outcome) if outcome is not None else output


_DATA_URL_RE = re.compile(
    r"^data:(?P<media_type>[^;,]+);base64,(?P<data>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def image_data_url(media_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def parse_image_data_url(value: str) -> tuple[str, str] | None:
    match = _DATA_URL_RE.match(value)
    if match is None:
        return None
    return match.group("media_type").lower(), match.group("data")


def responses_output_text(output: ToolOutput | Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        chunks: list[str] = []
        for block in output:
            if not isinstance(block, dict):
                continue
            typ = block.get("type")
            if typ in {"input_text", "text", "output_text"}:
                chunks.append(str(block.get("text") or ""))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(output or "")


# Only sessions written before outcome metadata require these text heuristics.
TOOL_FAILURE_PREFIXES: tuple[str, ...] = (
    "[error:",
    "[tool argument error:",
    "[unknown tool:",
    "[edit error:",
    "[edit conflict:",
    "[edit denied",
    "[command denied",
    "[read error:",
    "[read image unavailable:",
    "[web error:",
    "[tool disabled:",
    "[tool unavailable:",
    "[tool not parallel-safe:",
    "[finish requires",
    "[skipped:",
    "[not executed:",
    "[interactive command aborted:",
)


def tool_output_failed(output_text: str) -> bool:
    """Use execution metadata; text detection is only for legacy history."""
    outcome = tool_outcome(output_text)
    if outcome is not None:
        return outcome.status not in {"success", "running", "unknown"}
    return output_text.startswith(TOOL_FAILURE_PREFIXES) or (
        "cancelled by user" in output_text
    ) or bool(re.search(
        r"(?:^|\n)\[(?:exit code (?!0\])[-\d]+|timed out after [^\n]+)\](?:\n|$)",
        output_text,
    ))


def flatten_content_text(content: ToolOutput | Any) -> str:
    """Flatten structured content to display text.

    Handles both live tool-output blocks (``input_text``/``output_text``/
    ``input_image``) and persisted history blocks (``text`` /
    ``{"content": str}``).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    lines: list[str] = []
    image_count = 0
    for block in content:
        if not isinstance(block, dict):
            lines.append(str(block))
            continue
        typ = block.get("type")
        if typ in {"input_text", "text", "output_text"}:
            text = str(block.get("text") or "").strip()
            if text:
                lines.append(text)
            continue
        if typ == "input_image":
            parsed = parse_image_data_url(str(block.get("image_url") or ""))
            image_count += 1
            if parsed is None:
                lines.append(
                    f"[image output {image_count}: external or invalid image URL]"
                )
                continue
            media_type, data = parsed
            approx_bytes = (len(data) * 3) // 4
            lines.append(
                f"[image output {image_count}: {media_type}, {approx_bytes} bytes]"
            )
            continue
        if isinstance(block.get("content"), str):
            lines.append(block["content"])
            continue
        lines.append(f"[{typ or 'item'}]")
    return "\n".join(lines)


def summarize_tool_output(output: ToolOutput | Any) -> str:
    return preserve_tool_outcome(flatten_content_text(output), output)


def to_chat_tool_content(output: ToolOutput | Any) -> str | list[dict[str, Any]]:
    if not isinstance(output, list):
        return str(output or "")

    parts: list[dict[str, Any]] = []
    for block in output:
        if not isinstance(block, dict):
            continue
        typ = block.get("type")
        if typ in {"input_text", "text", "output_text"}:
            text = str(block.get("text") or "")
            if text:
                parts.append({"type": "text", "text": text})
        elif typ == "input_image":
            image_url = str(block.get("image_url") or "")
            if image_url:
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
    if parts:
        return parts
    return summarize_tool_output(output)


def to_anthropic_tool_result_content(output: ToolOutput | Any) -> str | list[dict[str, Any]]:
    if not isinstance(output, list):
        return str(output or "")

    blocks: list[dict[str, Any]] = []
    for block in output:
        if not isinstance(block, dict):
            continue
        typ = block.get("type")
        if typ in {"input_text", "text", "output_text"}:
            text = str(block.get("text") or "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif typ == "input_image":
            parsed = parse_image_data_url(str(block.get("image_url") or ""))
            if parsed is None:
                continue
            media_type, data = parsed
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            })
    if blocks:
        return blocks
    return summarize_tool_output(output)


def image_extension_for_media_type(media_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(media_type.lower(), ".img")


def to_gemini_function_response_parts(
    output: ToolOutput | Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(output, list):
        return {"result": output if output is not None else ""}, []

    text = responses_output_text(output)
    refs: list[dict[str, str]] = []
    parts: list[dict[str, Any]] = []
    image_index = 0
    for block in output:
        if not isinstance(block, dict) or block.get("type") != "input_image":
            continue
        parsed = parse_image_data_url(str(block.get("image_url") or ""))
        if parsed is None:
            continue
        media_type, data = parsed
        image_index += 1
        display_name = (
            f"read_image_{image_index}{image_extension_for_media_type(media_type)}"
        )
        refs.append({"$ref": display_name, "mimeType": media_type})
        parts.append({
            "inlineData": {
                "mimeType": media_type,
                "data": data,
            },
            "displayName": display_name,
        })

    response: dict[str, Any] = {"result": text}
    if refs:
        response["images"] = refs[0] if len(refs) == 1 else refs
    return response, parts
