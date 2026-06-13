"""Live provider model discovery and small Jarv-oriented recommendations."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import CONFIG_DIR
from .provider_catalog import FALLBACK_PROVIDER_MODELS, LOCAL_PROVIDERS


CACHE_DIR = CONFIG_DIR / "model-catalog"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_ENDPOINTS_DIR = "openrouter-endpoints"


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


def _openrouter_endpoints_path(model: str) -> Path:
    digest = hashlib.sha256(model.lower().encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / OPENROUTER_ENDPOINTS_DIR / f"{digest}.json"


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
                    "canonical_slug",
                    "default_parameters",
                    "expiration_date",
                    "pricing",
                    "supported_parameters",
                    "top_provider",
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
            metadata={
                key: item[key]
                for key in (
                    "capabilities",
                    "max_input_tokens",
                    "max_tokens",
                    "type",
                )
                if key in item
            },
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
                    "thinking",
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


def discover_openrouter_models(config: dict) -> list[CatalogModel]:
    """Fetch OpenRouter's public catalog for cross-provider pricing metadata."""
    timeout = float(config.get("model_catalog_timeout", 10))
    connect_timeout = float(config.get("model_catalog_connect_timeout", 5))
    with httpx.Client(
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
        headers={"User-Agent": "jarv"},
    ) as client:
        response = client.get(OPENROUTER_MODELS_URL)
        response.raise_for_status()
        return _normalize_openai_models(response.json())


def refresh_openrouter_pricing(config: dict) -> list[CatalogModel]:
    """Refresh OpenRouter pricing, falling back to its disk cache."""
    models: list[CatalogModel] = []
    try:
        models = discover_openrouter_models(config)
    except Exception:
        models = []
    if models:
        _write_cache("openrouter", models)
        return models
    return _read_cache("openrouter")


def discover_openrouter_endpoints(
    config: dict,
    model: str,
) -> list[dict[str, Any]]:
    """Fetch route-specific metadata for one OpenRouter model."""
    timeout = float(config.get("model_catalog_timeout", 10))
    connect_timeout = float(config.get("model_catalog_connect_timeout", 5))
    url = f"{OPENROUTER_MODELS_URL}/{model}/endpoints"
    with httpx.Client(
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
        headers={"User-Agent": "jarv"},
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(endpoints, list):
        return []
    return [item for item in endpoints if isinstance(item, dict)]


def _write_openrouter_endpoints(model: str, endpoints: list[dict]) -> None:
    try:
        path = _openrouter_endpoints_path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "fetched_at": time.time(),
            "endpoints": endpoints,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def cached_openrouter_endpoints(model: str | None) -> list[dict[str, Any]]:
    if not model:
        return []
    try:
        payload = json.loads(
            _openrouter_endpoints_path(str(model)).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return []
    if (
        not isinstance(payload, dict)
        or str(payload.get("model") or "").lower() != str(model).lower()
        or not isinstance(payload.get("endpoints"), list)
    ):
        return []
    return [
        item for item in payload["endpoints"]
        if isinstance(item, dict)
    ]


def refresh_openrouter_endpoints(config: dict, model: str) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    try:
        endpoints = discover_openrouter_endpoints(config, model)
    except Exception:
        endpoints = []
    if endpoints:
        _write_openrouter_endpoints(model, endpoints)
        return endpoints
    return cached_openrouter_endpoints(model)


_OPENROUTER_PROVIDER_NAMESPACES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "deepseek": "deepseek",
}


def _canonical_model_id(value: str) -> str:
    normalized = re.sub(r"(?<=\d)p(?=\d)", "", value.lower())
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _model_basename(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1]


def _without_snapshot(value: str) -> str:
    return re.sub(r"[-.]?\d{8}$", "", value)


def _model_family_key(value: str) -> str:
    basename = _model_basename(value).lower().split(":", 1)[0]
    basename = re.sub(r"(?<=\d)p(?=\d)", "", basename)
    basename = re.sub(r"\d+b[-.]?\d+e", "", basename)
    basename = re.sub(
        r"(?:^|[-.])(instruct|versatile|instant|fp8)(?=$|[-.])",
        "-",
        basename,
    )
    return _canonical_model_id(basename)


def _unique_preferred_match(models: list[CatalogModel]) -> CatalogModel | None:
    if len(models) == 1:
        return models[0]
    paid = [item for item in models if not item.id.endswith(":free")]
    if len(paid) == 1:
        return paid[0]
    return None


def resolve_openrouter_model(
    provider: str | None,
    model: str | None,
) -> CatalogModel | None:
    """Resolve a provider model ID to its OpenRouter catalog entry."""
    if not model:
        return None
    models = _read_cache("openrouter")
    if not models:
        return None

    model_id = str(model).strip()
    candidates = [model_id]
    namespace = _OPENROUTER_PROVIDER_NAMESPACES.get(str(provider or ""))
    if namespace and "/" not in model_id:
        candidates.append(f"{namespace}/{model_id}")
    snapshotless = _without_snapshot(model_id)
    if snapshotless != model_id:
        candidates.append(snapshotless)
        if namespace and "/" not in snapshotless:
            candidates.append(f"{namespace}/{snapshotless}")

    by_id = {item.id.lower(): item for item in models}
    for candidate in candidates:
        match = by_id.get(candidate.lower())
        if match is not None:
            return match

    by_canonical: dict[str, list[CatalogModel]] = {}
    for item in models:
        by_canonical.setdefault(_canonical_model_id(item.id), []).append(item)
    for candidate in candidates:
        matches = by_canonical.get(_canonical_model_id(candidate), [])
        if len(matches) == 1:
            return matches[0]

    basename = _canonical_model_id(_model_basename(snapshotless))
    basename_matches = [
        item
        for item in models
        if _canonical_model_id(_model_basename(item.id)) == basename
    ]
    match = _unique_preferred_match(basename_matches)
    if match is not None:
        return match

    family = _model_family_key(snapshotless)
    family_matches = [
        item for item in models if _model_family_key(item.id) == family
    ]
    return _unique_preferred_match(family_matches)


def cached_provider_model(
    config: dict,
    model: str | None = None,
) -> CatalogModel | None:
    """Return an exact model entry from the active provider cache."""
    provider = str(config.get("provider", "openai"))
    target = str(model or config.get("model") or "").strip().lower()
    if not target:
        return None
    return next(
        (
            item
            for item in _read_cache(provider)
            if item.id.lower() == target
        ),
        None,
    )


def openrouter_prices_for_model(
    provider: str | None,
    model: str | None,
) -> dict[str, float] | None:
    """Return OpenRouter catalog prices normalized to dollars per million tokens."""
    catalog_model = resolve_openrouter_model(provider, model)
    if catalog_model is None:
        return None
    pricing = catalog_model.metadata.get("pricing")
    if not isinstance(pricing, dict):
        return None
    try:
        prices = {
            "input": float(pricing["prompt"]) * 1_000_000,
            "output": float(pricing["completion"]) * 1_000_000,
        }
        cached_price = pricing.get("input_cache_read")
        if cached_price is not None:
            prices["cached_input"] = float(cached_price) * 1_000_000
    except (KeyError, TypeError, ValueError):
        return None
    if any(price < 0 for price in prices.values()):
        return None
    return prices


def _format_price_rate(value: float) -> str:
    if value >= 0.1:
        return f"${value:.2f}"
    if value >= 0.01:
        return f"${value:.3f}"
    return f"${value:.6f}".rstrip("0").rstrip(".")


def model_pricing_values(
    provider: str | None,
    model: str | None,
) -> tuple[str, str, str]:
    """Return display-ready input, cached-input, and output prices."""
    prices = openrouter_prices_for_model(provider, model)
    if prices is None:
        return "n/a", "n/a", "n/a"
    cached = (
        _format_price_rate(prices["cached_input"])
        if "cached_input" in prices
        else "n/a"
    )
    return (
        _format_price_rate(prices["input"]),
        cached,
        _format_price_rate(prices["output"]),
    )


def model_pricing_summary(provider: str | None, model: str | None) -> str:
    """Return compact OpenRouter reference pricing for a model picker row."""
    return " / ".join(model_pricing_values(provider, model))


def model_choice_description(
    provider: str | None,
    model: str,
    description: str,
) -> str:
    """Combine a recommendation label with its OpenRouter pricing reference."""
    pricing = model_pricing_summary(provider, model)
    return f"{pricing} | {description}" if description else pricing


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


def cached_provider_has_model(config: dict, model: str | None) -> bool:
    """Return whether a model is present in the active provider's cached catalog."""
    if not model:
        return False
    target = str(model).strip().lower()
    if not target:
        return False
    provider = str(config.get("provider", "openai"))
    return any(item.id.lower() == target for item in _read_cache(provider))


def cached_provider_model_ids(config: dict) -> list[str]:
    """Return model IDs from the active provider's cached catalog."""
    provider = str(config.get("provider", "openai"))
    return [item.id for item in _read_cache(provider)]


def refresh_model_choices(config: dict) -> list[tuple[str, str]]:
    """Refresh choices from the provider, falling back to cached data on failure."""
    provider = str(config.get("provider", "openai"))
    key = catalog_cache_key(config)
    refresh_openrouter_pricing(config)
    models: list[CatalogModel] = []
    try:
        models = discover_models(config)
    except Exception:
        models = []
    if models:
        _write_cache(provider, models)
    else:
        models = _read_cache(provider)
    if provider == "openrouter" and config.get("model"):
        refresh_openrouter_endpoints(config, str(config["model"]))

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
