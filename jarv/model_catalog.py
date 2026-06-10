"""Live provider model discovery and small Jarv-oriented recommendations."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import CONFIG_DIR
from .provider_catalog import FALLBACK_PROVIDER_MODELS, LOCAL_PROVIDERS


CACHE_DIR = CONFIG_DIR / "model-catalog"


@dataclass
class CatalogModel:
    id: str
    created: float = 0
    display_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_MEMORY_CHOICES: dict[str, list[tuple[str, str]]] = {}
_CACHE_LOCK = threading.RLock()


def catalog_cache_key(config: dict) -> str:
    provider = str(config.get("provider", "openai"))
    base_url = str(config.get("base_url") or "")
    return f"{provider}|{base_url}"


def _cache_path(provider: str) -> Path:
    safe_provider = re.sub(r"[^a-zA-Z0-9_.-]+", "_", provider)
    return CACHE_DIR / f"{safe_provider}.json"


def _write_cache(provider: str, models: list[CatalogModel]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": provider,
            "fetched_at": time.time(),
            "models": [asdict(model) for model in models],
        }
        path = _cache_path(provider)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def _read_cache(provider: str) -> list[CatalogModel]:
    try:
        payload = json.loads(_cache_path(provider).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    result: list[CatalogModel] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        result.append(CatalogModel(
            id=item["id"],
            created=_timestamp(item.get("created")),
            display_name=str(item.get("display_name") or ""),
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        ))
    return result


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0
    return 0


def _normalize_openai_models(payload: dict) -> list[CatalogModel]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        result.append(CatalogModel(
            id=item["id"],
            created=_timestamp(item.get("created")),
            display_name=str(item.get("name") or ""),
            metadata={
                key: item[key]
                for key in (
                    "architecture",
                    "context_length",
                    "pricing",
                    "supported_parameters",
                )
                if key in item
            },
        ))
    return result


def _normalize_anthropic_models(payload: dict) -> list[CatalogModel]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        result.append(CatalogModel(
            id=item["id"],
            created=_timestamp(item.get("created_at")),
            display_name=str(item.get("display_name") or ""),
        ))
    return result


def _normalize_gemini_models(payload: dict) -> list[CatalogModel]:
    data = payload.get("models")
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        methods = item.get("supportedGenerationMethods")
        if isinstance(methods, list) and "generateContent" not in methods:
            continue
        model_id = item["name"].removeprefix("models/")
        result.append(CatalogModel(
            id=model_id,
            display_name=str(item.get("displayName") or ""),
            metadata={
                key: item[key]
                for key in (
                    "inputTokenLimit",
                    "outputTokenLimit",
                    "supportedGenerationMethods",
                )
                if key in item
            },
        ))
    return result


def discover_models(config: dict) -> list[CatalogModel]:
    """Fetch and normalize the current provider's visible model catalog."""
    from .provider import create_client, get_backend, resolve_api_key

    provider = str(config.get("provider", "openai"))
    if provider not in LOCAL_PROVIDERS and not resolve_api_key(config):
        return []

    catalog_config = dict(config)
    timeout = float(config.get("model_catalog_timeout", 10))
    connect_timeout = float(config.get("model_catalog_connect_timeout", 5))
    catalog_config["http_timeout"] = timeout
    catalog_config["http_connect_timeout"] = connect_timeout
    catalog_config["anthropic_timeout"] = timeout
    catalog_config["anthropic_connect_timeout"] = connect_timeout
    catalog_config["gemini_timeout"] = timeout
    catalog_config["gemini_connect_timeout"] = connect_timeout

    client = create_client(catalog_config)
    try:
        backend = get_backend(catalog_config)
        if backend in ("responses", "openai_compat"):
            from .openai_http import list_models

            return _normalize_openai_models(list_models(client))
        if backend == "anthropic":
            from .anthropic_http import list_models

            return _normalize_anthropic_models(list_models(client))
        if backend == "gemini":
            from .gemini_http import list_models

            return _normalize_gemini_models(list_models(client))
        return []
    finally:
        client.close()


def _version_key(*parts: str | None) -> tuple[int, ...]:
    return tuple(int(part or 0) for part in parts)


def _choose_versions(
    models: list[CatalogModel],
    pattern: re.Pattern[str],
    tiers: list[tuple[str, str]],
    *,
    family_group: str = "family",
    version_groups: tuple[str, ...] = ("major", "minor"),
    snapshot_group: str | None = None,
) -> list[tuple[str, str]]:
    candidates: dict[str, list[tuple[tuple[int, ...], bool, CatalogModel]]] = {}
    for model in models:
        match = pattern.fullmatch(model.id)
        if not match:
            continue
        family = match.group(family_group)
        version = _version_key(*(match.group(group) for group in version_groups))
        stable = not snapshot_group or not match.group(snapshot_group)
        candidates.setdefault(family, []).append((version, stable, model))

    choices = []
    for family, description in tiers:
        family_models = candidates.get(family, [])
        if not family_models:
            continue
        _version, _stable, selected = max(
            family_models,
            key=lambda item: (item[0], item[1], item[2].created),
        )
        choices.append((selected.id, description))
    return choices


_OPENAI_PATTERN = re.compile(
    r"^gpt-(?P<major>\d+)\.(?P<minor>\d+)(?:-(?P<family>mini|nano))?$"
)
_ANTHROPIC_PATTERN = re.compile(
    r"^claude-(?P<family>fable|opus|sonnet|haiku)-"
    r"(?P<major>\d+)(?:-(?P<minor>\d{1,2}))?(?:-(?P<snapshot>\d{8}))?$"
)
_GEMINI_PATTERN = re.compile(
    r"^gemini-(?P<major>\d+)(?:\.(?P<minor>\d+))?-"
    r"(?P<family>flash-lite|pro|flash)"
    r"(?P<snapshot>(?:-preview)?(?:-\d{2}-\d{2})?)$"
)
_DEEPSEEK_PATTERN = re.compile(
    r"^deepseek-v(?P<major>\d+)(?:\.(?P<minor>\d+))?-(?P<family>pro|flash)$",
    re.IGNORECASE,
)


def _openai_choices(models: list[CatalogModel]) -> list[tuple[str, str]]:
    normalized = []
    for model in models:
        match = _OPENAI_PATTERN.fullmatch(model.id)
        if not match:
            continue
        family = match.group("family") or "flagship"
        normalized.append(CatalogModel(
            id=model.id,
            created=model.created,
            display_name=model.display_name,
            metadata={**model.metadata, "family": family},
        ))

    candidates: dict[str, list[tuple[tuple[int, int], CatalogModel]]] = {}
    for model in normalized:
        match = _OPENAI_PATTERN.fullmatch(model.id)
        if match:
            family = str(model.metadata["family"])
            version = (int(match.group("major")), int(match.group("minor")))
            candidates.setdefault(family, []).append((version, model))

    result = []
    for family, description in (
        ("flagship", "Flagship - latest GPT"),
        ("mini", "Balanced - latest GPT mini"),
        ("nano", "Budget - latest GPT nano"),
    ):
        if candidates.get(family):
            result.append((max(candidates[family], key=lambda item: item[0])[1].id, description))
    return result


def _anthropic_choices(models: list[CatalogModel]) -> list[tuple[str, str]]:
    return _choose_versions(
        models,
        _ANTHROPIC_PATTERN,
        [
            ("fable", "Premium - latest Claude Fable"),
            ("opus", "Flagship - latest Claude Opus"),
            ("sonnet", "Balanced - latest Claude Sonnet"),
            ("haiku", "Budget - latest Claude Haiku"),
        ],
        snapshot_group="snapshot",
    )


def _gemini_choices(models: list[CatalogModel]) -> list[tuple[str, str]]:
    return _choose_versions(
        models,
        _GEMINI_PATTERN,
        [
            ("pro", "Flagship - latest Gemini Pro"),
            ("flash", "Balanced - latest Gemini Flash"),
            ("flash-lite", "Budget - latest Gemini Flash-Lite"),
        ],
        snapshot_group="snapshot",
    )


def _deepseek_choices(models: list[CatalogModel]) -> list[tuple[str, str]]:
    return _choose_versions(
        models,
        _DEEPSEEK_PATTERN,
        [
            ("pro", "Flagship - latest DeepSeek Pro"),
            ("flash", "Budget - latest DeepSeek Flash"),
        ],
    )


def _latest_matching(
    models: list[CatalogModel],
    patterns: list[str],
) -> CatalogModel | None:
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = [model for model in models if regex.search(model.id)]
        if matches:
            return max(matches, key=lambda model: (model.created, _numbers(model.id), model.id))
    return None


def _numbers(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


_FAMILY_POLICIES: dict[str, list[tuple[str, str, list[str]]]] = {
    "openrouter": [
        (
            "Premium",
            "latest Claude Fable",
            [r"^anthropic/claude-fable-\d+(?:[.-]\d+)?$"],
        ),
        (
            "Flagship",
            "latest Claude Opus",
            [r"^anthropic/claude-opus-\d+(?:[.-]\d+)?$"],
        ),
        (
            "Balanced",
            "latest Claude Sonnet",
            [r"^anthropic/claude-sonnet-\d+(?:[.-]\d+)?$"],
        ),
        (
            "Budget",
            "latest fast budget model",
            [
                r"^anthropic/claude-haiku-\d+(?:[.-]\d+)?$",
                r"^google/gemini-\d+(?:\.\d+)?-flash-lite(?:-preview)?$",
                r":free$",
            ],
        ),
    ],
    "groq": [
        ("Flagship", "latest large general model", [r"gpt-oss-120b", r"llama.*70b"]),
        ("Balanced", "latest versatile model", [r"llama.*versatile", r"qwen.*32b"]),
        ("Budget", "latest instant model", [r"llama.*8b.*instant", r"instant"]),
    ],
    "together": [
        ("Flagship", "latest DeepSeek Pro", [r"deepseek.*v\d+.*pro", r"deepseek"]),
        ("Balanced", "latest Llama Maverick", [r"llama.*maverick", r"llama.*instruct"]),
        ("Budget", "latest small Qwen", [r"qwen.*(?:7b|8b|9b)", r"qwen"]),
    ],
    "fireworks": [
        ("Flagship", "latest Kimi", [r"kimi"]),
        ("Balanced", "latest MiniMax", [r"minimax"]),
        ("Budget", "latest small Qwen", [r"qwen.*(?:7b|8b|9b)", r"qwen"]),
    ],
}

_DEFAULT_TIERS = {
    "openai": "balanced",
    "openrouter": "balanced",
    "anthropic": "balanced",
    "gemini": "balanced",
    "groq": "flagship",
    "deepseek": "budget",
    "together": "balanced",
    "fireworks": "flagship",
}


def _family_policy_choices(
    provider: str,
    models: list[CatalogModel],
) -> list[tuple[str, str]]:
    result = []
    seen: set[str] = set()
    for tier, label, patterns in _FAMILY_POLICIES.get(provider, []):
        selected = _latest_matching(models, patterns)
        if selected is None or selected.id in seen:
            continue
        seen.add(selected.id)
        result.append((selected.id, f"{tier} - {label}"))
    return result


def recommend_models(provider: str, models: list[CatalogModel]) -> list[tuple[str, str]]:
    """Reduce a provider catalog to the models Jarv users are likely to want."""
    policies: dict[str, Callable[[list[CatalogModel]], list[tuple[str, str]]]] = {
        "openai": _openai_choices,
        "anthropic": _anthropic_choices,
        "gemini": _gemini_choices,
        "deepseek": _deepseek_choices,
    }
    if provider in LOCAL_PROVIDERS:
        return [(model.id, "Installed model") for model in sorted(models, key=lambda item: item.id.lower())]
    if provider in policies:
        return policies[provider](models)
    return _family_policy_choices(provider, models)


def _merge_fallbacks(
    provider: str,
    choices: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    if choices:
        return choices
    return list(FALLBACK_PROVIDER_MODELS.get(provider, []))


def get_model_choices(
    config: dict,
    *,
    refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return live recommendations, using cache and bundled fallbacks on failure."""
    if not refresh:
        return get_cached_model_choices(config)
    return refresh_model_choices(config)


def get_cached_model_choices(config: dict) -> list[tuple[str, str]]:
    """Return choices without network access, using memory, disk, then fallbacks."""
    provider = str(config.get("provider", "openai"))
    key = catalog_cache_key(config)
    with _CACHE_LOCK:
        if key in _MEMORY_CHOICES:
            return list(_MEMORY_CHOICES[key])

    models = _read_cache(provider)
    choices = _merge_fallbacks(provider, recommend_models(provider, models))
    with _CACHE_LOCK:
        _MEMORY_CHOICES[key] = list(choices)
    return list(choices)


def refresh_model_choices(config: dict) -> list[tuple[str, str]]:
    """Refresh choices from the provider, falling back to cached data on failure."""
    provider = str(config.get("provider", "openai"))
    key = catalog_cache_key(config)
    models: list[CatalogModel] = []
    try:
        models = discover_models(config)
    except Exception:
        models = []
    if models:
        _write_cache(provider, models)
    else:
        models = _read_cache(provider)

    choices = _merge_fallbacks(provider, recommend_models(provider, models))
    with _CACHE_LOCK:
        _MEMORY_CHOICES[key] = list(choices)
    return list(choices)


def get_default_model(
    config: dict,
    *,
    choices: list[tuple[str, str]] | None = None,
) -> str:
    """Choose the preferred live tier when a new provider is selected."""
    provider = str(config.get("provider", "openai"))
    available = choices if choices is not None else get_model_choices(config)
    preferred_tier = _DEFAULT_TIERS.get(provider)
    if preferred_tier:
        for model, description in available:
            if description.lower().startswith(preferred_tier):
                return model
    if available:
        return available[0][0]
    return str(config.get("model") or "local-model")


def clear_memory_cache() -> None:
    """Clear process-local choices. Intended for tests and explicit refreshes."""
    with _CACHE_LOCK:
        _MEMORY_CHOICES.clear()
