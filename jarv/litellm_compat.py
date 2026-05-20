"""Compatibility helpers for importing LiteLLM."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType


_FILTER_INSTALLED = False
_OPTIONAL_PRELOAD_WARNINGS = (
    "could not pre-load bedrock-runtime response stream shape",
    "could not pre-load sagemaker-runtime response stream shape",
)


class _OptionalPreloadFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(warning in message for warning in _OPTIONAL_PRELOAD_WARNINGS)


def import_litellm() -> ModuleType:
    """Import LiteLLM without noisy optional AWS stream-shape warnings."""
    global _FILTER_INSTALLED
    if not _FILTER_INSTALLED:
        logging.getLogger("LiteLLM").addFilter(_OptionalPreloadFilter())
        _FILTER_INSTALLED = True
    return importlib.import_module("litellm")
