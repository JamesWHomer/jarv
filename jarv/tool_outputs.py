from __future__ import annotations

import base64
import re
from typing import Any, TypeAlias


ToolOutput: TypeAlias = str | list[dict[str, Any]]

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


TOOL_FAILURE_PREFIXES: tuple[str, ...] = (
    "[error:",
    "[tool argument error:",
    "[unknown tool:",
    "[edit error:",
    "[edit denied",
    "[read error:",
    "[read image unavailable:",
    "[web error:",
    "[tool disabled:",
)


def tool_output_failed(output_text: str) -> bool:
    """Single owner of the "did this tool call fail?" rule.

    All dispatch layers encode failures as bracketed output prefixes, and the
    output string is exactly what session history persists — so this detector
    works identically for live cards and history re-renders.
    """
    return output_text.startswith(TOOL_FAILURE_PREFIXES) or (
        "cancelled by user" in output_text
    )


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
    return flatten_content_text(output)


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
