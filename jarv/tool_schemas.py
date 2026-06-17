"""Provider-specific normalization for Jarv tool schemas."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(schema)
    typ = out.get("type")
    if isinstance(typ, list):
        if "null" not in typ:
            out["type"] = [*typ, "null"]
    elif isinstance(typ, str):
        out["type"] = [typ, "null"]
    elif "anyOf" in out:
        any_of = out.get("anyOf")
        if isinstance(any_of, list) and not any(
            isinstance(item, dict) and item.get("type") == "null"
            for item in any_of
        ):
            out["anyOf"] = [*any_of, {"type": "null"}]
    else:
        out["type"] = ["string", "null"]
    return out


def _openai_strict_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return deepcopy(schema)

    out = deepcopy(schema)
    typ = out.get("type")

    for key in ("anyOf", "oneOf", "allOf"):
        variants = out.get(key)
        if isinstance(variants, list):
            out[key] = [_openai_strict_schema(item) for item in variants]

    if typ == "array" or "items" in out:
        out["items"] = _openai_strict_schema(out.get("items", {}))

    if typ == "object" or "properties" in out:
        properties = out.get("properties")
        if isinstance(properties, dict):
            original_required = set(out.get("required") or [])
            strict_properties = {}
            for name, property_schema in properties.items():
                normalized = _openai_strict_schema(property_schema)
                if name not in original_required and isinstance(normalized, dict):
                    normalized = _nullable(normalized)
                strict_properties[name] = normalized
            out["properties"] = strict_properties
            out["required"] = list(strict_properties)
        else:
            out["properties"] = {}
            out["required"] = []
        out["additionalProperties"] = False

    return out


def strict_openai_tools(tools: list[dict]) -> list[dict]:
    """Return tools in OpenAI strict schema form without mutating callers."""
    strict_tools = []
    for tool in tools:
        copied = deepcopy(tool)
        function = copied.get("function")
        if (
            copied.get("type") == "function"
            and isinstance(function, dict)
        ):
            function["parameters"] = _openai_strict_schema(
                function.get("parameters", {"type": "object"})
            )
            function["strict"] = True
            function.pop("input_examples", None)
        elif copied.get("type") == "function":
            copied["parameters"] = _openai_strict_schema(
                copied.get("parameters", {"type": "object"})
            )
            copied["strict"] = True
            copied.pop("input_examples", None)
        strict_tools.append(copied)
    return strict_tools
