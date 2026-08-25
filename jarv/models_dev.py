"""models.dev facts for the models Jarv can talk to.

Jarv learns *which* models a provider exposes from the provider itself, in
:func:`jarv.model_catalog.discover_models`. This module answers *what* those
models are: prices, context limits, input modalities, and reasoning controls.

The data is models.dev, an MIT-licensed community catalog. A pruned snapshot
ships inside the package so a fresh install is useful offline, and
:func:`refresh` layers a newer copy into the config directory with a
conditional GET, so revalidating an unchanged catalog costs no bytes.

Cost is only ever read from the entry of the provider actually serving the
model. A model sold by several providers has one set of capabilities but many
different prices, so non-cost facts fall back to another provider's entry while
prices never do.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .paths import CONFIG_DIR
from .provider_catalog import models_dev_provider, models_dev_provider_ids


DEFAULT_URL = "https://models.dev/api.json"
SNAPSHOT_PATH = Path(__file__).parent / "data" / "models_dev.json"
CACHE_PATH = CONFIG_DIR / "models-dev.json"

PROVIDER_FIELDS = frozenset({"id", "name", "env", "doc", "api"})
MODEL_FIELDS = frozenset({
    "id",
    "name",
    "family",
    "attachment",
    "reasoning",
    "reasoning_options",
    "tool_call",
    "structured_output",
    "temperature",
    "interleaved",
    "knowledge",
    "release_date",
    "last_updated",
    "modalities",
    "open_weights",
    "limit",
    "cost",
})

# OpenRouter namespaces the models it resells, so looking up an OpenRouter fact
# for a bare provider model ID needs the namespace put back on.
_NAMESPACES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "deepseek": "deepseek",
    "groq": "groq",
}

_LOCK = threading.RLock()
_CATALOG: dict[str, dict[str, Any]] | None = None
_STAMPS: tuple[Any, ...] = ()
_INDEXES: dict[str, dict[str, dict[str, list[str]]]] = {}


@dataclass(frozen=True)
class ModelFacts:
    """One model as models.dev describes it."""

    provider: str
    id: str
    native: bool = True
    name: str = ""
    family: str = ""
    attachment: bool | None = None
    reasoning: bool | None = None
    reasoning_options: tuple[dict[str, Any], ...] = ()
    tool_call: bool | None = None
    structured_output: bool | None = None
    temperature: bool | None = None
    interleaved: bool | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    context_limit: int | None = None
    output_limit: int | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    release_date: str = ""
    knowledge: str = ""
    open_weights: bool | None = None

    def rates(self, input_tokens: int | None = None) -> dict[str, float] | None:
        """Prices in dollars per million tokens, or None when not sold here."""
        if not self.native:
            return None
        return _rates(self.cost, input_tokens)


# ---------------------------------------------------------------------------
# Model ID normalization
# ---------------------------------------------------------------------------

def canonical_model_id(value: str) -> str:
    """Collapse a model ID to comparable alphanumerics: kimi-k2p6 -> kimik26."""
    normalized = re.sub(r"(?<=\d)p(?=\d)", "", value.lower())
    return re.sub(r"[^a-z0-9]+", "", normalized)


def model_basename(value: str) -> str:
    """Drop any namespace: accounts/fireworks/models/kimi-k3 -> kimi-k3."""
    return value.rstrip("/").rsplit("/", 1)[-1]


def without_snapshot(value: str) -> str:
    """Drop a trailing date stamp: claude-haiku-4-5-20251001 -> claude-haiku-4-5."""
    return re.sub(r"[-.]?\d{8}$", "", value)


def model_family_key(value: str) -> str:
    """Collapse serving variants of one model onto a shared key."""
    basename = model_basename(value).lower().split(":", 1)[0]
    basename = re.sub(r"(?<=\d)p(?=\d)", "", basename)
    basename = re.sub(r"\d+b[-.]?\d+e", "", basename)
    basename = re.sub(
        r"(?:^|[-.])(instruct|versatile|instant|fp8)(?=$|[-.])",
        "-",
        basename,
    )
    return canonical_model_id(basename)


def family_stem(value: str) -> str:
    """Strip version numbers so gpt-5.7-mini lines up with the gpt-mini family."""
    basename = model_basename(without_snapshot(value)).lower().split(":", 1)[0]
    return canonical_model_id(re.sub(r"[0-9]+", "", basename))


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def prune_catalog(payload: Any, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """Keep only the providers Jarv talks to and the fields Jarv reads."""
    if not isinstance(payload, dict):
        return {}
    providers: dict[str, dict[str, Any]] = {}
    for provider_id, entry in payload.items():
        if provider_id not in wanted or not isinstance(entry, dict):
            continue
        models = entry.get("models")
        if not isinstance(models, dict):
            continue
        kept = {
            model_id: {key: value for key, value in model.items() if key in MODEL_FIELDS}
            for model_id, model in models.items()
            if isinstance(model_id, str) and isinstance(model, dict)
        }
        if not kept:
            continue
        pruned = {key: value for key, value in entry.items() if key in PROVIDER_FIELDS}
        pruned["models"] = kept
        providers[provider_id] = pruned
    return providers


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _providers_of(payload: dict | None) -> dict[str, Any]:
    providers = (payload or {}).get("providers")
    return providers if isinstance(providers, dict) else {}


def catalog() -> dict[str, dict[str, Any]]:
    """Return provider ID to provider entry, the snapshot overlaid by any refresh."""
    global _CATALOG, _STAMPS
    stamps = (_stamp(SNAPSHOT_PATH), _stamp(CACHE_PATH))
    with _LOCK:
        if _CATALOG is not None and stamps == _STAMPS:
            return _CATALOG

    merged: dict[str, dict[str, Any]] = {}
    for provider_id, entry in _providers_of(_read_json(SNAPSHOT_PATH)).items():
        if isinstance(entry, dict) and isinstance(entry.get("models"), dict):
            merged[provider_id] = entry
    for provider_id, entry in _providers_of(_read_json(CACHE_PATH)).items():
        if isinstance(entry, dict) and entry.get("models"):
            merged[provider_id] = entry

    with _LOCK:
        _CATALOG = merged
        _STAMPS = stamps
        _INDEXES.clear()
    return merged


def clear_cache() -> None:
    """Drop the parsed catalog. Intended for tests and explicit refreshes."""
    global _CATALOG, _STAMPS
    with _LOCK:
        _CATALOG = None
        _STAMPS = ()
        _INDEXES.clear()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def _index(provider_id: str, models: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    with _LOCK:
        cached = _INDEXES.get(provider_id)
        if cached is not None:
            return cached

    built: dict[str, dict[str, list[str]]] = {
        "exact": {},
        "canonical": {},
        "basename": {},
        "family": {},
    }
    for model_id in models:
        built["exact"].setdefault(model_id.lower(), []).append(model_id)
        built["canonical"].setdefault(canonical_model_id(model_id), []).append(model_id)
        built["basename"].setdefault(
            canonical_model_id(model_basename(without_snapshot(model_id))), []
        ).append(model_id)
        built["family"].setdefault(model_family_key(model_id), []).append(model_id)

    with _LOCK:
        _INDEXES[provider_id] = built
    return built


def _preferred(ids: list[str]) -> str | None:
    """Pick a single ID from an ambiguous match, preferring one paid entry."""
    if len(ids) == 1:
        return ids[0]
    paid = [model_id for model_id in ids if not model_id.endswith(":free")]
    if len(paid) == 1:
        return paid[0]
    return None


def _candidates(provider_id: str, source_provider: str | None, model_id: str) -> list[str]:
    values = [model_id]
    stripped = without_snapshot(model_id)
    if stripped != model_id:
        values.append(stripped)
    namespace = _NAMESPACES.get(str(source_provider or ""))
    if provider_id == "openrouter" and namespace and "/" not in model_id:
        values.extend(f"{namespace}/{value}" for value in tuple(values))
    return list(dict.fromkeys(values))


def _probe(bucket: str, candidate: str) -> str:
    if bucket == "canonical":
        return canonical_model_id(candidate)
    if bucket == "basename":
        return canonical_model_id(model_basename(without_snapshot(candidate)))
    return model_family_key(candidate)


def _find(
    data: dict[str, Any],
    provider_id: str,
    source_provider: str | None,
    model_id: str,
) -> str | None:
    entry = data.get(provider_id)
    models = entry.get("models") if isinstance(entry, dict) else None
    if not isinstance(models, dict):
        return None
    index = _index(provider_id, models)
    candidates = _candidates(provider_id, source_provider, model_id)

    for candidate in candidates:
        found = index["exact"].get(candidate.lower())
        if found:
            return found[0]
    for bucket in ("canonical", "basename", "family"):
        for candidate in candidates:
            match = _preferred(index[bucket].get(_probe(bucket, candidate), []))
            if match is not None:
                return match
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip().lower() for item in value if isinstance(item, str))


def _optional_bool(raw: dict[str, Any], key: str) -> bool | None:
    value = raw.get(key)
    return value if isinstance(value, bool) else None


def _facts(provider_id: str, model_id: str, raw: dict[str, Any], native: bool) -> ModelFacts:
    limit = raw.get("limit") if isinstance(raw.get("limit"), dict) else {}
    modalities = raw.get("modalities") if isinstance(raw.get("modalities"), dict) else {}
    options = raw.get("reasoning_options")
    return ModelFacts(
        provider=provider_id,
        id=model_id,
        native=native,
        name=str(raw.get("name") or ""),
        family=str(raw.get("family") or ""),
        attachment=_optional_bool(raw, "attachment"),
        reasoning=_optional_bool(raw, "reasoning"),
        reasoning_options=tuple(
            option
            for option in (options if isinstance(options, list) else ())
            if isinstance(option, dict)
        ),
        tool_call=_optional_bool(raw, "tool_call"),
        structured_output=_optional_bool(raw, "structured_output"),
        temperature=_optional_bool(raw, "temperature"),
        interleaved=_optional_bool(raw, "interleaved"),
        input_modalities=_strings(modalities.get("input")),
        output_modalities=_strings(modalities.get("output")),
        context_limit=_positive_int(limit.get("context")),
        output_limit=_positive_int(limit.get("output")),
        cost=raw.get("cost") if isinstance(raw.get("cost"), dict) else {},
        release_date=str(raw.get("release_date") or ""),
        knowledge=str(raw.get("knowledge") or ""),
        open_weights=_optional_bool(raw, "open_weights"),
    )


def lookup(provider: str | None, model: str | None) -> ModelFacts | None:
    """Return facts for a model, preferring the provider that actually serves it.

    A match found under a different provider comes back with ``native=False``:
    its capabilities describe the same model, but its prices belong to another
    seller, so :meth:`ModelFacts.rates` withholds them.
    """
    model_id = str(model or "").strip()
    if not model_id:
        return None
    data = catalog()
    provider_id = models_dev_provider(provider)

    order: list[str] = []
    if provider_id and provider_id in data:
        order.append(provider_id)
    for other in ("openrouter", *sorted(data)):
        if other in data and other not in order:
            order.append(other)

    for candidate_provider in order:
        found = _find(data, candidate_provider, provider, model_id)
        if found is None:
            continue
        raw = data[candidate_provider]["models"][found]
        return _facts(candidate_provider, found, raw, candidate_provider == provider_id)
    return None


def nearest_family(provider: str | None, model: str | None) -> ModelFacts | None:
    """Newest catalog entry in a model's family, for IDs newer than the catalog.

    For capability defaults only. A model released after the last refresh still
    behaves like its family, but it has its own price, so the result is never
    native and never yields rates.
    """
    model_id = str(model or "").strip()
    provider_id = models_dev_provider(provider)
    if not model_id or not provider_id:
        return None
    entry = catalog().get(provider_id)
    models = entry.get("models") if isinstance(entry, dict) else None
    if not isinstance(models, dict):
        return None

    stem = family_stem(model_id)
    if not stem:
        return None
    matches = [
        (str(raw.get("release_date") or ""), model_key, raw)
        for model_key, raw in models.items()
        if isinstance(raw, dict)
        and canonical_model_id(str(raw.get("family") or "")) == stem
    ]
    if not matches:
        return None
    _date, model_key, raw = max(matches, key=lambda item: (item[0], item[1]))
    return replace(_facts(provider_id, model_key, raw, True), native=False)


def models_for(provider: str | None) -> list[ModelFacts]:
    """Every catalog entry for a provider, newest first."""
    provider_id = models_dev_provider(provider)
    entry = catalog().get(provider_id) if provider_id else None
    models = entry.get("models") if isinstance(entry, dict) else None
    if not isinstance(models, dict) or provider_id is None:
        return []
    facts = [
        _facts(provider_id, model_id, raw, True)
        for model_id, raw in models.items()
        if isinstance(raw, dict)
    ]
    facts.sort(key=lambda item: (item.release_date, item.id), reverse=True)
    return facts


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def _tier_cost(cost: dict[str, Any], input_tokens: int | None) -> dict[str, Any]:
    """Apply the long-context surcharge a request of this size actually pays."""
    if not input_tokens or input_tokens <= 0:
        return cost
    best: tuple[float, dict[str, Any]] | None = None
    tiers = cost.get("tiers")
    for tier in tiers if isinstance(tiers, list) else ():
        if not isinstance(tier, dict):
            continue
        bounds = tier.get("tier")
        if not isinstance(bounds, dict) or bounds.get("type") != "context":
            continue
        size = bounds.get("size")
        if not isinstance(size, (int, float)) or input_tokens <= size:
            continue
        if best is None or size > best[0]:
            best = (float(size), tier)
    if best is not None:
        return {**cost, **{key: value for key, value in best[1].items() if key != "tier"}}
    legacy = cost.get("context_over_200k")
    if isinstance(legacy, dict) and input_tokens > 200_000:
        return {**cost, **legacy}
    return cost


def _rates(cost: Any, input_tokens: int | None = None) -> dict[str, float] | None:
    if not isinstance(cost, dict):
        return None
    effective = _tier_cost(cost, input_tokens)
    try:
        rates = {
            "input": float(effective["input"]),
            "output": float(effective["output"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    for source, target in (("cache_read", "cached_input"), ("cache_write", "cache_write")):
        value = effective.get(source)
        if value is None:
            continue
        try:
            rates[target] = float(value)
        except (TypeError, ValueError):
            continue
    if any(rate < 0 for rate in rates.values()):
        return None
    return rates


def prices(
    provider: str | None,
    model: str | None,
    *,
    input_tokens: int | None = None,
) -> dict[str, float] | None:
    """Dollars per million tokens for a model, as this provider sells it."""
    facts = lookup(provider, model)
    if facts is None:
        return None
    return facts.rates(input_tokens)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

def _cached_etag() -> str:
    payload = _read_json(CACHE_PATH)
    if not _providers_of(payload):
        return ""
    return str((payload or {}).get("etag") or "")


def _write_cache(providers: dict[str, Any], etag: str, source: str) -> bool:
    from datetime import datetime, timezone

    payload = {
        "source": source,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "etag": etag,
        "providers": providers,
    }
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(CACHE_PATH)
    except OSError:
        return False
    clear_cache()
    return True


def refresh(config: dict | None = None) -> bool:
    """Revalidate the catalog against models.dev. True when the cache is current."""
    import httpx

    config = config or {}
    url = str(config.get("models_dev_url") or DEFAULT_URL)
    timeout = float(config.get("model_catalog_timeout", 10) or 10)
    connect_timeout = float(config.get("model_catalog_connect_timeout", 5) or 5)

    headers = {"User-Agent": "jarv"}
    etag = _cached_etag()
    if etag:
        headers["If-None-Match"] = etag

    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            follow_redirects=True,
        ) as client:
            response = client.get(url, headers=headers)
            if response.status_code == 304:
                return True
            response.raise_for_status()
            payload = response.json()
            response_etag = response.headers.get("ETag", "")
    except Exception:
        return False

    providers = prune_catalog(payload, models_dev_provider_ids())
    if not providers:
        return False
    return _write_cache(providers, response_etag, url)
