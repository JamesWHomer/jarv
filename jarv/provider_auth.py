"""Provider authentication helpers without transport dependencies."""

import os

from .provider_catalog import LOCAL_PROVIDERS, PROVIDERS


def resolve_api_key(config: dict) -> str:
    provider_name = config.get("provider", "openai")
    per_provider = config.get("api_keys", {}).get(provider_name, "")
    if per_provider:
        return per_provider
    key = config.get("api_key", "")
    if key:
        return key
    info = PROVIDERS.get(provider_name, {})
    env_key = info.get("env_key")
    if env_key:
        return os.environ.get(env_key, "")
    if provider_name in LOCAL_PROVIDERS:
        return "not-needed"
    return ""
