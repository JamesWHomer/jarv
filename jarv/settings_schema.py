"""Settings row schema derived from config_schema."""

from __future__ import annotations

from .config_schema import CONFIG_FIELDS, SETTINGS_TOOL_LABELS, TOOL_NAMES


def settings_service_tier_choices(config: dict) -> tuple[tuple[str, str], ...]:
    from .provider_catalog import service_tier_choices

    provider = str(config.get("provider", "openai"))
    return tuple((tier, tier) for tier in service_tier_choices(provider))


def settings_service_tier_description(config: dict) -> str:
    provider = str(config.get("provider", "openai"))
    if provider == "anthropic":
        return "priority uses committed capacity, then falls back to standard"
    if len(settings_service_tier_choices(config)) == 1:
        return "this provider uses standard processing"
    return "standard cost, flex savings, or priority latency"


def _field_to_row(field, config: dict) -> dict:
    row = {
        "section": field.section,
        "label": field.label,
        "key": field.key,
        "kind": field.ui_kind,
        "desc": field.desc,
    }
    if field.empty:
        row["empty"] = field.empty
    if field.multiline:
        row["multiline"] = True
    if field.ui_kind == "choice":
        if field.key == "reasoning_effort":
            from .reasoning import reasoning_effort_choices, reasoning_effort_description

            row["choices"] = reasoning_effort_choices(config)
            row["desc"] = reasoning_effort_description(config)
        elif field.settings_choices:
            row["choices"] = field.settings_choices
    if field.ui_kind == "setup":
        if field.key == "provider":
            row["step"] = "provider"
        elif field.key == "api_key":
            row["step"] = "key"
        elif field.key == "model":
            row["step"] = "model"
    return row


def settings_rows(config: dict) -> list[dict]:
    rows = [_field_to_row(field, config) for field in CONFIG_FIELDS if field.ui_kind]
    tool_rows = [
        {
            "section": "tools",
            "label": SETTINGS_TOOL_LABELS[name][0],
            "key": f"tool:{name}",
            "tool_name": name,
            "kind": "tool_bool",
            "desc": SETTINGS_TOOL_LABELS[name][1],
        }
        for name in TOOL_NAMES
    ]
    insert_at = next(
        (index for index, row in enumerate(rows) if row["section"] == "tools"),
        len(rows),
    )
    if insert_at == len(rows):
        rows.extend(tool_rows)
    else:
        rows[insert_at:insert_at] = tool_rows

    tier_choices = settings_service_tier_choices(config)
    if len(tier_choices) > 1:
        api_key_index = next(
            (index for index, row in enumerate(rows) if row.get("key") == "api_key"),
            1,
        )
        rows.insert(
            api_key_index + 1,
            {
                "section": "account",
                "label": "Processing tier",
                "key": "service_tier",
                "kind": "choice",
                "choices": tier_choices,
                "desc": settings_service_tier_description(config),
            },
        )
    return rows
