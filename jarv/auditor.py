"""Command auditor — uses an LLM to decide whether a flagged command is safe.

When `audited` mode is enabled, flagged commands are sent to an LLM auditor
instead of immediately prompting the user. The auditor sees the command, the
risk classification, and a brief context summary. It either approves (command
runs automatically with a printed reason) or defers to the user (showing why
it recommends caution).
"""

import json
import re

from .display import console
from .provider import resolve_api_key, PROVIDERS, LOCAL_PROVIDERS


AUDITOR_SYSTEM_PROMPT = """\
You are a command safety auditor for a CLI assistant. Your job is to decide \
whether a flagged shell command is safe to auto-execute given the context.

You will receive:
- The command that was flagged
- The risk category (why it was flagged)
- A brief context summary (what the user/agent is trying to accomplish)

Respond with a JSON object (no markdown fencing):
{"allow": true/false, "reason": "short one-sentence explanation"}

Guidelines:
- ALLOW commands that are clearly safe in context (e.g., `rm -rf node_modules` \
during a clean build, `git reset --hard` on an unmodified working tree, \
`pip install requests --user`).
- DENY (allow=false) commands that could cause irreversible damage, data loss, \
or security issues that the context doesn't justify. When denying, your reason \
should explain what makes you cautious.
- Be pragmatic. Most flagged commands are routine development operations that \
happen to match a broad pattern. Lean toward allowing unless genuinely risky.
- Keep your reason under 15 words.\
"""

AUDITOR_RETRY_INSTRUCTION = """\
Your previous response could not be parsed.
Return only valid JSON matching this schema:
{"allow": true/false, "reason": "short one-sentence explanation"}
Do not include markdown, code fences, or extra text.\
"""


def _get_auditor_model(config: dict) -> str:
    """Return the configured auditor model, or the active model."""
    auditor_model = config.get("auditor_model", "")
    if auditor_model:
        return auditor_model
    return config.get("model", "gpt-4.1-mini")


def _build_context_summary(history: list, max_chars: int = 600) -> str:
    """Extract a short context summary from recent history.

    Pulls the last user message and last assistant text to give the auditor
    a sense of what's happening without sending the full conversation.
    """
    last_user = ""
    last_assistant = ""

    for item in reversed(history):
        role = item.get("role", "")
        content = item.get("content", "") or ""
        if role == "user" and not last_user:
            last_user = content[:300]
        elif role == "assistant" and not last_assistant:
            last_assistant = content[:300]
        if last_user and last_assistant:
            break

    parts = []
    if last_user:
        parts.append(f"User asked: {last_user}")
    if last_assistant:
        parts.append(f"Assistant said: {last_assistant}")

    summary = "\n".join(parts)
    return summary[:max_chars] if summary else "(no context available)"


def audit_command(
    command: str,
    reason: str,
    config: dict,
    history: list | None = None,
) -> tuple[bool, str]:
    """Run the auditor on a flagged command.

    Returns (allow, reason_text).
    - allow=True: command should auto-execute
    - allow=False: command should be shown to user for manual confirmation
    """
    context_summary = _build_context_summary(history or [])

    user_message = (
        f"Command: {command}\n"
        f"Risk category: {reason}\n"
        f"Context: {context_summary}"
    )

    model = _get_auditor_model(config)
    provider = config.get("provider", "openai")
    info = PROVIDERS.get(provider, {})
    backend = info.get("backend", "openai_compat")

    try:
        if backend == "litellm":
            return _call_litellm(config, model, user_message)
        else:
            return _call_openai_compat(config, model, user_message, info)
    except Exception as e:
        # If auditor fails, fall back to user prompt
        return False, f"auditor unavailable ({type(e).__name__})"


def _call_openai_compat(
    config: dict, model: str, user_message: str, info: dict
) -> tuple[bool, str]:
    from openai import OpenAI

    api_key = resolve_api_key(config)
    base_url = config.get("base_url")
    if not base_url:
        base_url = info.get("base_url")

    kwargs = {"api_key": api_key or "not-needed"}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)

    kwargs = _openai_compat_kwargs(
        config,
        model,
        info,
        _auditor_messages(user_message),
    )
    response = client.chat.completions.create(**kwargs)
    parsed = _parse_response(response.choices[0].message.content or "")
    if not _is_parse_failure(parsed):
        return parsed

    retry_kwargs = _openai_compat_kwargs(
        config,
        model,
        info,
        _auditor_messages(user_message, retry=True),
    )
    retry_response = client.chat.completions.create(**retry_kwargs)
    return _parse_response(retry_response.choices[0].message.content or "")


def _auditor_messages(user_message: str, *, retry: bool = False) -> list[dict[str, str]]:
    """Build the auditor prompt, optionally with a strict retry instruction."""
    content = user_message
    if retry:
        content = f"{user_message}\n\n{AUDITOR_RETRY_INSTRUCTION}"
    return [
        {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _openai_compat_kwargs(
    config: dict,
    model: str,
    info: dict,
    messages: list[dict[str, str]],
) -> dict:
    kwargs = {
        "model": model,
        "messages": messages,
    }
    if _uses_max_completion_tokens(config, model, info):
        kwargs["max_completion_tokens"] = 300
    else:
        kwargs["temperature"] = 0
        kwargs["max_tokens"] = 100

    return kwargs


def _uses_max_completion_tokens(config: dict, model: str, info: dict) -> bool:
    """Return True for OpenAI Chat models that reject max_tokens."""
    if not _is_direct_openai(config, info):
        return False
    model = model.lower()
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _is_direct_openai(config: dict, info: dict) -> bool:
    """Return True when the OpenAI SDK is talking to OpenAI directly."""
    provider = config.get("provider", "openai")
    return provider == "openai" and not (config.get("base_url") or info.get("base_url"))


def _call_litellm(
    config: dict, model: str, user_message: str
) -> tuple[bool, str]:
    from .litellm_compat import import_litellm

    litellm = import_litellm()

    provider_name = config.get("provider", "")
    prefix = PROVIDERS.get(provider_name, {}).get("litellm_prefix", "")
    litellm_model = f"{prefix}/{model}" if prefix and "/" not in model else model

    kwargs = _litellm_kwargs(litellm_model, _auditor_messages(user_message))

    api_key = resolve_api_key(config)
    if api_key and api_key != "not-needed":
        kwargs["api_key"] = api_key

    response = litellm.completion(**kwargs)
    parsed = _parse_response(response.choices[0].message.content or "")
    if not _is_parse_failure(parsed):
        return parsed

    retry_kwargs = _litellm_kwargs(litellm_model, _auditor_messages(user_message, retry=True))
    if api_key and api_key != "not-needed":
        retry_kwargs["api_key"] = api_key
    retry_response = litellm.completion(**retry_kwargs)
    return _parse_response(retry_response.choices[0].message.content or "")


def _litellm_kwargs(litellm_model: str, messages: list[dict[str, str]]) -> dict:
    return {
        "model": litellm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 100,
    }


def _is_parse_failure(parsed: tuple[bool, str]) -> bool:
    return parsed == (False, "could not parse auditor response")


def _parse_response(text: str) -> tuple[bool, str]:
    """Parse the auditor's response.

    JSON is preferred, but some models return a short prose verdict despite
    being prompted for JSON. Accept clear allow/deny responses so the auditor
    can still make a useful decision.
    """
    text = text.strip()
    parsed_json = _parse_json_verdict(text)
    if parsed_json:
        return parsed_json

    parsed = _parse_loose_verdict(text)
    if parsed:
        return parsed

    return False, "could not parse auditor response"


def _parse_json_verdict(text: str) -> tuple[bool, str] | None:
    """Return the first valid JSON verdict object found in arbitrary text."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", text):
        try:
            data, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue

        parsed = _coerce_json_verdict(data)
        if parsed:
            return parsed
    return None


def _coerce_json_verdict(data) -> tuple[bool, str] | None:
    """Validate and normalize an auditor JSON object."""
    if not isinstance(data, dict):
        return None

    allow = _coerce_json_bool(data["allow"]) if "allow" in data else _coerce_json_alias(data)
    if allow is None:
        return None

    raw_reason = data.get("reason")
    reason = str(raw_reason).strip() if raw_reason is not None else ""
    return allow, reason or "no reason given"


def _coerce_json_bool(value) -> bool | None:
    """Accept booleans and clear boolean-like strings only."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "y"):
            return True
        if normalized in ("false", "no", "n"):
            return False
    return None


def _coerce_json_alias(data: dict) -> bool | None:
    """Accept clear schema-ish verdict aliases from non-strict providers."""
    for key, allow_values, deny_values in (
        ("verdict", ("allow", "allowed"), ("deny", "denied")),
        ("decision", ("approve", "approved", "allow", "allowed"), ("reject", "rejected", "deny", "denied")),
    ):
        value = data.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        if normalized in allow_values:
            return True
        if normalized in deny_values:
            return False
    return None


def _parse_loose_verdict(text: str) -> tuple[bool, str] | None:
    """Accept common non-JSON verdicts returned by weaker JSON followers."""
    normalized = text.strip()
    if not normalized:
        return None

    lowered = normalized.lower()

    allow_match = re.search(r"\ballow(?:ed)?\b\s*[:=-]?\s*(true|yes|y)\b", lowered)
    deny_match = re.search(r"\b(?:allow(?:ed)?\b\s*[:=-]?\s*(false|no|n)|deny|denied)\b", lowered)
    leading_allow = re.match(r"^\s*(?:verdict\s*[:=-]\s*)?(allow|approved|safe)\b", lowered)
    leading_deny = re.match(r"^\s*(?:verdict\s*[:=-]\s*)?(deny|denied|reject|rejected|unsafe)\b", lowered)

    if allow_match or leading_allow:
        return True, _loose_reason(normalized, "auditor allowed command")
    if deny_match or leading_deny:
        return False, _loose_reason(normalized, "auditor denied command")
    return None


def _loose_reason(text: str, fallback: str) -> str:
    """Extract a concise reason from non-JSON auditor text."""
    reason_match = re.search(r"\breason\s*[:=-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else text
    reason = re.sub(r"^\s*(?:verdict\s*[:=-]\s*)?(?:allow(?:ed)?|approved|safe|deny|denied|reject(?:ed)?|unsafe)\b\s*[:=-]?\s*", "", reason, flags=re.IGNORECASE)
    reason = " ".join(reason.split())
    return reason[:120] if reason else fallback
