"""Model-aware reasoning capability resolution and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")
CONFIGURED_EFFORTS = ("", "none", *EFFORT_LEVELS)


@dataclass(frozen=True)
class ReasoningCapabilities:
    supported: bool | None = None
    efforts: tuple[str, ...] | None = None
    modes: tuple[str, ...] | None = None
    supports_disable: bool | None = None
    returns_reasoning: bool | None = None
    native_effort: bool | None = None
    max_output_tokens: int | None = None
    sources: dict[str, str] = field(default_factory=dict)


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _supported_children(value: Any, names: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, dict):
        return ()
    return tuple(
        name
        for name in names
        if isinstance(value.get(name), dict)
        and value[name].get("supported") is True
    )


def _static_capabilities(provider: str, model: str) -> ReasoningCapabilities:
    lowered = model.lower()
    source = {"policy": "built-in model policy"}

    if provider == "openrouter":
        return ReasoningCapabilities(
            efforts=("minimal", "low", "medium", "high", "xhigh"),
            supports_disable=True,
            native_effort=True,
            sources=source,
        )

    if provider == "openai":
        if lowered.startswith("gpt-4o"):
            return ReasoningCapabilities(
                supported=False,
                efforts=(),
                supports_disable=False,
                native_effort=False,
                sources=source,
            )
        if "gpt-5" not in lowered:
            return ReasoningCapabilities(sources=source)
        if "pro" in lowered:
            return ReasoningCapabilities(
                supported=True,
                efforts=("high",),
                supports_disable=False,
                native_effort=True,
                sources=source,
            )
        if any(version in lowered for version in ("gpt-5.4", "gpt-5.5")):
            return ReasoningCapabilities(
                supported=True,
                efforts=("low", "medium", "high", "xhigh"),
                supports_disable=True,
                native_effort=True,
                sources=source,
            )
        return ReasoningCapabilities(
            supported=True,
            efforts=("minimal", "low", "medium", "high"),
            supports_disable="gpt-5.1" in lowered or "gpt-5.2" in lowered,
            native_effort=True,
            sources=source,
        )

    if provider == "anthropic":
        if any(name in lowered for name in ("fable-5", "opus-4-8", "opus-4.8", "opus-4-7", "opus-4.7")):
            return ReasoningCapabilities(
                supported=True,
                efforts=("low", "medium", "high", "xhigh", "max"),
                modes=("adaptive",),
                supports_disable=False,
                native_effort=True,
                sources=source,
            )
        if any(version in lowered for version in ("-4-6", "-4.6")):
            return ReasoningCapabilities(
                supported=True,
                efforts=("low", "medium", "high", "max"),
                modes=("enabled", "adaptive"),
                supports_disable=True,
                native_effort=True,
                sources=source,
            )
        if "opus-4-5" in lowered or "opus-4.5" in lowered:
            return ReasoningCapabilities(
                supported=True,
                efforts=("low", "medium", "high"),
                modes=("enabled",),
                supports_disable=True,
                native_effort=True,
                sources=source,
            )
        if "claude" in lowered:
            return ReasoningCapabilities(
                supported=True,
                efforts=EFFORT_LEVELS,
                modes=("enabled",),
                supports_disable=True,
                native_effort=False,
                sources=source,
            )

    if provider == "gemini":
        if lowered.startswith("gemini-3"):
            efforts = (
                ("low", "medium", "high")
                if "pro" in lowered
                else ("minimal", "low", "medium", "high")
            )
            return ReasoningCapabilities(
                supported=True,
                efforts=efforts,
                modes=("level",),
                supports_disable=False,
                native_effort=True,
                sources=source,
            )
        if lowered.startswith("gemini-2.5"):
            return ReasoningCapabilities(
                supported=True,
                efforts=("minimal", "low", "medium", "high"),
                modes=("budget",),
                supports_disable="pro" not in lowered,
                native_effort=False,
                sources=source,
            )

    if provider == "deepseek" and "deepseek" in lowered:
        return ReasoningCapabilities(
            supported=True,
            efforts=("high", "max"),
            modes=("enabled",),
            supports_disable=True,
            native_effort=True,
            sources=source,
        )

    if provider == "groq":
        if "gpt-oss" in lowered:
            return ReasoningCapabilities(
                supported=True,
                efforts=("low", "medium", "high"),
                supports_disable=False,
                native_effort=True,
                sources=source,
            )
        if "qwen3" in lowered:
            return ReasoningCapabilities(
                supported=True,
                efforts=(),
                supports_disable=True,
                native_effort=False,
                sources=source,
            )

    return ReasoningCapabilities(sources=source)


def _openrouter_metadata_capabilities(metadata: dict[str, Any]) -> ReasoningCapabilities:
    parameters = metadata.get("supported_parameters")
    parameters_advertised = isinstance(parameters, list)
    supported_parameters = (
        {str(value) for value in parameters}
        if parameters_advertised
        else set()
    )
    supported = (
        "reasoning" in supported_parameters
        if parameters_advertised
        else None
    )
    returns_reasoning = (
        "include_reasoning" in supported_parameters
        if parameters_advertised
        else None
    )
    top_provider = metadata.get("top_provider")
    top_provider = top_provider if isinstance(top_provider, dict) else {}
    max_output = _positive_int(
        top_provider.get("max_completion_tokens")
        or metadata.get("max_completion_tokens")
    )
    return ReasoningCapabilities(
        supported=supported,
        returns_reasoning=returns_reasoning,
        max_output_tokens=max_output,
        sources={
            key: "OpenRouter model catalog"
            for key, value in (
                ("supported", supported),
                ("returns_reasoning", returns_reasoning),
                ("max_output_tokens", max_output),
            )
            if value is not None
        },
    )


def _anthropic_metadata_capabilities(metadata: dict[str, Any]) -> ReasoningCapabilities:
    capabilities = metadata.get("capabilities")
    if not isinstance(capabilities, dict):
        return ReasoningCapabilities()

    thinking = capabilities.get("thinking")
    thinking = thinking if isinstance(thinking, dict) else {}
    supported = _bool(thinking.get("supported"))
    modes = _supported_children(
        thinking.get("types"),
        ("enabled", "adaptive"),
    )

    effort = capabilities.get("effort")
    effort = effort if isinstance(effort, dict) else {}
    native_effort = _bool(effort.get("supported"))
    efforts = _supported_children(effort, EFFORT_LEVELS)
    if supported and not efforts and "enabled" in modes:
        efforts = EFFORT_LEVELS

    supports_disable = None
    if supported is True:
        supports_disable = "enabled" in modes

    max_output = _positive_int(metadata.get("max_tokens"))
    source = "Anthropic Models API"
    return ReasoningCapabilities(
        supported=supported,
        efforts=efforts if supported else (),
        modes=modes or None,
        supports_disable=supports_disable,
        returns_reasoning=supported,
        native_effort=native_effort,
        max_output_tokens=max_output,
        sources={
            key: source
            for key, value in (
                ("supported", supported),
                ("efforts", efforts if supported else ()),
                ("modes", modes or None),
                ("supports_disable", supports_disable),
                ("native_effort", native_effort),
                ("max_output_tokens", max_output),
            )
            if value is not None
        },
    )


def _gemini_metadata_capabilities(metadata: dict[str, Any]) -> ReasoningCapabilities:
    supported = _bool(metadata.get("thinking"))
    max_output = _positive_int(metadata.get("outputTokenLimit"))
    source = "Gemini Models API"
    return ReasoningCapabilities(
        supported=supported,
        returns_reasoning=supported,
        max_output_tokens=max_output,
        sources={
            key: source
            for key, value in (
                ("supported", supported),
                ("returns_reasoning", supported),
                ("max_output_tokens", max_output),
            )
            if value is not None
        },
    )


def _merge(
    base: ReasoningCapabilities,
    override: ReasoningCapabilities,
) -> ReasoningCapabilities:
    values = {}
    for name in (
        "supported",
        "efforts",
        "modes",
        "supports_disable",
        "returns_reasoning",
        "native_effort",
        "max_output_tokens",
    ):
        replacement = getattr(override, name)
        values[name] = replacement if replacement is not None else getattr(base, name)
    values["sources"] = {**base.sources, **override.sources}
    return ReasoningCapabilities(**values)


def get_reasoning_capabilities(
    config: dict,
    model: str | None = None,
) -> ReasoningCapabilities:
    from .model_catalog import (
        cached_openrouter_endpoints,
        cached_provider_model,
        resolve_openrouter_model,
    )

    provider = str(config.get("provider", "openai"))
    selected_model = str(model or config.get("model") or "")
    result = _static_capabilities(provider, selected_model)

    openrouter_model = resolve_openrouter_model(provider, selected_model)
    if openrouter_model is not None:
        result = _merge(
            result,
            _openrouter_metadata_capabilities(openrouter_model.metadata),
        )

    native_model = cached_provider_model(config, selected_model)
    if native_model is not None:
        metadata = native_model.metadata
        if provider == "anthropic":
            result = _merge(result, _anthropic_metadata_capabilities(metadata))
        elif provider == "gemini":
            result = _merge(result, _gemini_metadata_capabilities(metadata))
        else:
            result = _merge(
                result,
                _openrouter_metadata_capabilities(metadata),
            )

    if provider == "openrouter":
        endpoints = cached_openrouter_endpoints(selected_model)
        if endpoints:
            reasoning_endpoints = [
                endpoint
                for endpoint in endpoints
                if "reasoning" in endpoint.get("supported_parameters", [])
            ]
            output_limits = [
                value
                for endpoint in reasoning_endpoints
                if (value := _positive_int(endpoint.get("max_completion_tokens")))
                is not None
            ]
            endpoint_capabilities = ReasoningCapabilities(
                supported=bool(reasoning_endpoints),
                returns_reasoning=(
                    any(
                        "include_reasoning"
                        in endpoint.get("supported_parameters", [])
                        for endpoint in reasoning_endpoints
                    )
                    if reasoning_endpoints
                    else False
                ),
                max_output_tokens=min(output_limits) if output_limits else None,
                sources={
                    "supported": "OpenRouter endpoint catalog",
                    "returns_reasoning": "OpenRouter endpoint catalog",
                    **(
                        {"max_output_tokens": "OpenRouter endpoint catalog"}
                        if output_limits
                        else {}
                    ),
                },
            )
            result = _merge(result, endpoint_capabilities)

    return result


def reasoning_effort_choices(config: dict) -> tuple[tuple[str, str], ...]:
    capabilities = get_reasoning_capabilities(config)
    choices: list[tuple[str, str]] = [("", "default")]
    if capabilities.supported is False:
        return tuple(choices)
    if capabilities.supports_disable is True:
        choices.append(("none", "none"))
    if capabilities.efforts is not None:
        choices.extend((effort, effort) for effort in capabilities.efforts)
    return tuple(choices)


def reasoning_effort_description(config: dict) -> str:
    capabilities = get_reasoning_capabilities(config)
    if capabilities.supported is False:
        return "selected model does not expose reasoning controls"
    if capabilities.efforts:
        return ", ".join(capabilities.efforts)
    if capabilities.supported is True:
        return "reasoning supported; effort levels are not advertised"
    return "default; model capability is unknown"


def reasoning_effort_error(
    config: dict,
    effort: Any | None = None,
) -> str | None:
    value = config.get("reasoning_effort", "") if effort is None else effort
    if value is None:
        value = ""
    if not isinstance(value, str):
        return "reasoning_effort must be a string"
    normalized = value.strip().lower()
    if normalized not in CONFIGURED_EFFORTS:
        return (
            "reasoning_effort must be one of: "
            + ", ".join("default" if item == "" else item for item in CONFIGURED_EFFORTS)
        )
    if not normalized:
        return None

    capabilities = get_reasoning_capabilities(config)
    model = str(config.get("model") or "selected model")
    if capabilities.supported is False:
        return f"{model} does not support reasoning controls"
    if normalized == "none":
        if capabilities.supports_disable is False:
            return f"{model} does not support disabling reasoning"
        return None
    if (
        capabilities.efforts is not None
        and normalized not in capabilities.efforts
    ):
        allowed = ", ".join(capabilities.efforts) or "default"
        return f"{model} supports reasoning efforts: {allowed}"
    return None


def reconcile_reasoning_effort(config: dict) -> str | None:
    """Reset a definitively unsupported effort and return the old value."""
    value = config.get("reasoning_effort", "")
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None

    capabilities = get_reasoning_capabilities(config)
    unsupported = capabilities.supported is False
    if normalized == "none":
        unsupported = unsupported or capabilities.supports_disable is False
    elif capabilities.efforts is not None:
        unsupported = unsupported or normalized not in capabilities.efforts

    if not unsupported:
        return None
    config["reasoning_effort"] = ""
    return normalized


def require_reasoning_effort(config: dict, effort: str) -> str:
    error = reasoning_effort_error(config, effort)
    if error:
        raise ValueError(error)
    return effort.strip().lower()
