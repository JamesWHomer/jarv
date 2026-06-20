"""Registry for LLM provider HTTP backends."""

from __future__ import annotations

from typing import Any

from .provider_auth import resolve_api_key
from .provider_catalog import PROVIDERS


class ProviderError(Exception):
    pass


def get_backend(config: dict) -> str:
    provider_name = config.get("provider", "openai")
    info = PROVIDERS.get(provider_name)
    if info:
        return info["backend"]
    if config.get("base_url"):
        return "openai_compat"
    return "responses"


def _http_module(backend: str):
    if backend in ("responses", "openai_compat"):
        from . import openai_http

        return openai_http
    if backend == "anthropic":
        from . import anthropic_http

        return anthropic_http
    if backend == "gemini":
        from . import gemini_http

        return gemini_http
    raise ProviderError(f"Unknown backend: {backend}")


def create_client(config: dict):
    backend = get_backend(config)
    api_key = resolve_api_key(config)
    module = _http_module(backend)
    if backend in ("responses", "openai_compat"):
        base_url = config.get("base_url")
        if not base_url:
            provider_name = config.get("provider", "openai")
            info = PROVIDERS.get(provider_name, {})
            base_url = info.get("base_url")
        return module.create_client(config, api_key, base_url)
    return module.create_client(config, api_key)


def list_models(client, backend: str, **kwargs: Any) -> dict:
    module = _http_module(backend)
    return module.list_models(client, **kwargs)
