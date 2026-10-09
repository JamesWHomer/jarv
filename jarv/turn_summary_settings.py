"""Shared definitions and migration for the configurable completed-turn summary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnSummaryField:
    name: str
    key: str
    label: str
    default: bool
    description: str


TURN_SUMMARY_FIELDS: tuple[TurnSummaryField, ...] = (
    TurnSummaryField("tokens", "turn_summary_tokens", "Token counts", True, "input, output, and total tokens across the turn's model requests"),
    TurnSummaryField("cache", "turn_summary_cache", "Cached tokens", True, "provider-reported cached input tokens across the turn"),
    TurnSummaryField("reasoning", "turn_summary_reasoning", "Reasoning tokens", True, "provider-reported reasoning output tokens across the turn"),
    TurnSummaryField("speed", "turn_summary_speed", "Output speed", True, "final response generation tok/s, from server timing or a stream estimate"),
    TurnSummaryField("time", "turn_summary_time", "Model time", True, "total model request time for the turn, excluding tool execution"),
    TurnSummaryField("session", "turn_summary_session", "Session tokens", False, "running token total for the saved session"),
    TurnSummaryField("cost", "turn_summary_cost", "Session cost", False, "running session cost, labelled when estimated or incomplete"),
)

_LEGACY_KEYS = ("print_usage_after_model", "print_usage_after_agent")


def migrate_turn_summary_settings(config: dict) -> bool:
    """Replace legacy display switches, preserving explicit canonical values.

    Validate all legacy values before changing anything. Call before adding
    schema defaults, so an absent value is distinguishable from an explicit one.
    """
    present = [key for key in _LEGACY_KEYS if key in config]
    if not present:
        return False
    for key in present:
        if not isinstance(config[key], bool):
            raise ValueError(f"Config '{key}' must be a boolean (true or false).")

    model_stats = config.get("print_usage_after_model", False)
    agent_usage = config.get("print_usage_after_agent", False)
    if model_stats or agent_usage:
        config.setdefault("turn_summary", True)
        legacy_fields = {
            "tokens": True,
            "cache": True,
            "reasoning": model_stats,
            "speed": model_stats,
            "time": model_stats,
            "session": agent_usage,
            "cost": agent_usage,
        }
        for field in TURN_SUMMARY_FIELDS:
            config.setdefault(field.key, legacy_fields[field.name])

    for key in present:
        del config[key]
    return True


def enabled_turn_summary_fields(config: dict) -> frozenset[str]:
    """Resolve selected fields without mutating direct or loaded configs."""
    configured = dict(config)
    migrate_turn_summary_settings(configured)
    if configured.get("turn_summary", False) is not True:
        return frozenset()
    return frozenset(
        field.name for field in TURN_SUMMARY_FIELDS
        if configured.get(field.key, field.default) is True
    )
