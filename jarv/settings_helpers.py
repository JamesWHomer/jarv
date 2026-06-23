"""Shared, dependency-light helpers for the settings command and its submodules.

Kept in a leaf module so settings_command (controller/editor) and
settings_model_picker can both depend on these without an import cycle.
"""

from .tui_layout import clip_text


AUDITOR_DEFAULT_MODEL_CHOICE = "default"


def _clip_text(value: str, width: int) -> str:
    return clip_text(value, width, ellipsis="\u2026")


def _settings_choice_label(value, choices: tuple[tuple[str, str], ...]) -> str:
    for key, label in choices:
        if value == key:
            return label
    return str(value)
