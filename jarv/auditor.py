"""Command auditor — uses an LLM to decide whether a flagged command is safe.

When `audited` mode is enabled, flagged commands are sent to an LLM auditor
instead of immediately prompting the user. The auditor sees the command, the
risk classification, and a brief context summary. It either approves (command
runs automatically with a printed reason) or defers to the user (showing why
it recommends caution).
"""

import json

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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=100,
    )

    return _parse_response(response.choices[0].message.content or "")


def _call_litellm(
    config: dict, model: str, user_message: str
) -> tuple[bool, str]:
    import litellm

    provider_name = config.get("provider", "")
    prefix = PROVIDERS.get(provider_name, {}).get("litellm_prefix", "")
    litellm_model = f"{prefix}/{model}" if prefix and "/" not in model else model

    kwargs = {
        "model": litellm_model,
        "messages": [
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_tokens": 100,
    }

    api_key = resolve_api_key(config)
    if api_key and api_key != "not-needed":
        kwargs["api_key"] = api_key

    response = litellm.completion(**kwargs)
    return _parse_response(response.choices[0].message.content or "")


def _parse_response(text: str) -> tuple[bool, str]:
    """Parse the auditor's JSON response."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
        allow = bool(data.get("allow", False))
        reason = str(data.get("reason", "no reason given"))
        return allow, reason
    except (json.JSONDecodeError, TypeError):
        return False, "could not parse auditor response"
