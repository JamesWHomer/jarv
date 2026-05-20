import os
import re
import sys

from rich.prompt import Prompt
from rich.text import Text

from .display import console, jarv_panel, section_rule
from .provider import PROVIDERS, LOCAL_PROVIDERS, KEY_PATTERNS


class GoBack(Exception):
    pass


PROVIDER_CHOICES = [
    ("openai", "OpenAI", "gpt-5.4-mini"),
    ("openrouter", "OpenRouter (200+ models)", "tencent/hy3-preview"),
    ("anthropic", "Anthropic", "claude-sonnet-4-6"),
    ("gemini", "Google Gemini", "gemini-3-flash-preview"),
    ("groq", "Groq", "openai/gpt-oss-120b"),
    ("deepseek", "DeepSeek", "deepseek-v4-flash"),
    ("together", "Together AI", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
    ("fireworks", "Fireworks AI", "accounts/fireworks/models/kimi-k2p6"),
    ("ollama", "Ollama", "llama3.3"),
    ("lm_studio", "LM Studio", "local-model"),
    ("vllm", "vLLM", "local-model"),
]

PROVIDER_MODELS = {
    "openai": [
        ("gpt-5.5", "Flagship — largest, smartest"),
        ("gpt-5.4-mini", "Balanced — faster, cheaper"),
        ("gpt-5.4-nano", "Budget — smallest, fastest"),
    ],
    "anthropic": [
        ("claude-opus-4-7", "Flagship — most capable"),
        ("claude-sonnet-4-6", "Balanced — fast and capable"),
        ("claude-haiku-4-5", "Budget — fastest, cheapest"),
    ],
    "openrouter": [
        # Top 15 by weekly token usage (openrouter.ai/models?order=top-weekly)
        ("tencent/hy3-preview", "Tencent — Hunyuan H3 (Hy3) Preview"),
        ("deepseek/deepseek-v4-flash", "DeepSeek — V4 Flash"),
        ("anthropic/claude-opus-4.7", "Anthropic — Claude Opus 4.7"),
        ("anthropic/claude-sonnet-4.6", "Anthropic — Claude Sonnet 4.6"),
        ("moonshotai/kimi-k2.6", "MoonshotAI — Kimi K2.6"),
        ("google/gemini-3-flash-preview", "Google — Gemini 3 Flash Preview"),
        ("deepseek/deepseek-v3.2", "DeepSeek — V3.2"),
        ("deepseek/deepseek-v4-pro", "DeepSeek — V4 Pro"),
        ("minimax/minimax-m2.7", "MiniMax — M2.7"),
        ("openrouter/owl-alpha", "OpenRouter — Owl Alpha"),
        ("stepfun/step-3.5-flash", "StepFun — Step 3.5 Flash"),
        ("nvidia/nemotron-3-super-120b-a12b:free", "NVIDIA — Nemotron 3 Super"),
        ("anthropic/claude-opus-4.6", "Anthropic — Claude Opus 4.6"),
        ("google/gemini-2.5-flash-lite", "Google — Gemini 2.5 Flash Lite"),
        ("google/gemini-2.5-flash", "Google — Gemini 2.5 Flash"),
    ],
    "gemini": [
        ("gemini-3.1-pro-preview", "Flagship — Gemini 3.1 Pro, 2M context"),
        ("gemini-3-flash-preview", "Balanced — Gemini 3 Flash"),
        ("gemini-3.1-flash-lite", "Budget — fastest, cheapest"),
    ],
    "groq": [
        ("openai/gpt-oss-120b", "Flagship — GPT OSS 120B"),
        ("llama-3.3-70b-versatile", "Balanced — Llama 3.3 70B"),
        ("llama-3.1-8b-instant", "Budget — fastest inference"),
    ],
    "deepseek": [
        ("deepseek-v4-pro", "Flagship — DeepSeek V4 Pro, 1M context"),
        ("deepseek-v4-flash", "Budget — faster, cheaper"),
    ],
    "together": [
        ("deepseek-ai/DeepSeek-V4-Pro", "Flagship — DeepSeek V4 Pro"),
        ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "Balanced — Llama 4 Maverick, 1M context"),
        ("Qwen/Qwen3.5-9B", "Budget — Qwen 3.5 9B"),
    ],
    "fireworks": [
        ("accounts/fireworks/models/kimi-k2p6", "Flagship — Kimi K2.6"),
        ("accounts/fireworks/models/minimax-m2p7", "Balanced — MiniMax M2.7"),
        ("accounts/fireworks/models/qwen3-8b", "Budget — Qwen3 8B"),
    ],
}

SETUP_STEPS = {"provider", "key", "model", "safety", "base_url"}
TOTAL_STEPS = 5


# ---------------------------------------------------------------------------
# Shell / env helpers
# ---------------------------------------------------------------------------

def _detect_shell_and_profile() -> tuple[str, str, str]:
    if sys.platform == "win32":
        return ("PowerShell", 'setx {env_key} "your-key-here"', "$PROFILE")
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return ("zsh", 'export {env_key}="your-key-here"', "~/.zshrc")
    elif "fish" in shell:
        return ("fish", 'set -Ux {env_key} "your-key-here"', "~/.config/fish/config.fish")
    else:
        return ("bash", 'export {env_key}="your-key-here"', "~/.bashrc")


def _show_env_instructions(provider_name: str) -> None:
    info = PROVIDERS.get(provider_name, {})
    env_key = info.get("env_key", "API_KEY")
    key_url = info.get("key_url", "")
    label = info.get("label", provider_name)
    shell_name, export_template, profile_path = _detect_shell_and_profile()
    export_cmd = export_template.format(env_key=env_key)

    console.print()
    if key_url:
        console.print(f"  [bold]1.[/bold] Get a key at [cyan]{key_url}[/cyan]")
    else:
        console.print(f"  [bold]1.[/bold] Get an API key from {label}")
    console.print(f"  [bold]2.[/bold] Add this to [bold]{profile_path}[/bold]:")
    console.print()
    console.print(f"     [bold green]{export_cmd}[/bold green]")
    console.print()
    console.print(f"  [bold]3.[/bold] Reload your shell and run [bold cyan]jarv /setup[/bold cyan] again")
    console.print()


# ---------------------------------------------------------------------------
# Individual setup steps
# ---------------------------------------------------------------------------

def setup_provider(config: dict) -> dict:
    console.print()
    console.print(section_rule("Provider", step=1, total=TOTAL_STEPS))
    console.print()

    cloud_providers = [(i, k, l) for i, (k, l, _) in enumerate(PROVIDER_CHOICES, 1) if k not in LOCAL_PROVIDERS]
    local_providers = [(i, k, l) for i, (k, l, _) in enumerate(PROVIDER_CHOICES, 1) if k in LOCAL_PROVIDERS]

    for i, key, label in cloud_providers:
        default_tag = " [green]← default[/green]" if i == 1 else ""
        console.print(f"  [bold cyan]{i:>2}.[/bold cyan] {label}{default_tag}")

    console.print()
    console.print("  [dim]── Local ──[/dim]")
    for i, key, label in local_providers:
        console.print(f"  [bold cyan]{i:>2}.[/bold cyan] {label}")
    console.print()

    while True:
        choice = Prompt.ask(
            "  [bold]Pick a provider[/bold] [dim](number or name, b=back)[/dim]",
            default="1",
            console=console,
        ).strip()

        if choice.lower() in ("b", "back"):
            raise GoBack()

        provider_name = _resolve_provider(choice)
        if provider_name is not None:
            break
        console.print(f"  [red]Unknown provider '{choice}'. Please pick again.[/red]")

    config["provider"] = provider_name
    return config


def setup_api_key(config: dict) -> dict:
    from .provider import resolve_api_key

    provider_name = config.get("provider", "openai")

    console.print()
    console.print(section_rule("API Key", step=2, total=TOTAL_STEPS))

    if provider_name in LOCAL_PROVIDERS:
        console.print(f"\n  [green]✓[/green] No API key needed for {PROVIDERS[provider_name]['label']}.")
        return config

    env_key_name = PROVIDERS.get(provider_name, {}).get("env_key", "")
    api_key = resolve_api_key(config)
    if api_key:
        masked = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
        if env_key_name and os.environ.get(env_key_name, ""):
            source = f"from {env_key_name}"
        elif config.get("api_keys", {}).get(provider_name):
            source = "per-provider config"
        else:
            source = "config"
        console.print(f"\n  [green]✓[/green] Found key [dim]{masked}[/dim] [dim italic]({source})[/dim italic]")
        console.print()
        action = Prompt.ask(
            "  [bold]Use existing key or enter a new one?[/bold] [dim](b=back)[/dim]",
            choices=["use", "overwrite", "b"],
            default="use",
            console=console,
        )
        if action == "b":
            raise GoBack()
        if action == "overwrite":
            api_key = _prompt_api_key(provider_name)
            config.setdefault("api_keys", {})[provider_name] = api_key
    else:
        label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
        console.print(f"\n  [yellow]![/yellow] No API key found for {label}.")
        console.print()
        api_key = _prompt_api_key(provider_name)
        config.setdefault("api_keys", {})[provider_name] = api_key

    return config


def setup_model(config: dict) -> dict:
    provider_name = config.get("provider", "openai")

    console.print()
    console.print(section_rule("Model", step=3, total=TOTAL_STEPS))
    console.print()

    models = PROVIDER_MODELS.get(provider_name)
    if models:
        for i, (name, desc) in enumerate(models, 1):
            default_tag = " [green]← default[/green]" if i == 1 else ""
            console.print(f"  [bold cyan]{i}.[/bold cyan] [bold]{name}[/bold] [dim]— {desc}[/dim]{default_tag}")
        console.print()

        while True:
            model_choice = Prompt.ask(
                "  [bold]Pick a model[/bold] [dim](number or name, b=back)[/dim]",
                default="1",
                console=console,
            ).strip()
            if model_choice.lower() in ("b", "back"):
                raise GoBack()
            model = _resolve_model(provider_name, model_choice)
            if model is not None:
                break
            try:
                int(model_choice)
                console.print(f"  [red]Invalid number. Please pick again.[/red]")
                continue
            except ValueError:
                pass
            console.print(f"  [yellow]Model '{model_choice}' not found.[/yellow] Are you sure it's correct?")
            confirm = Prompt.ask("  ", choices=["continue", "retry"], default="retry", console=console)
            if confirm == "continue":
                model = model_choice
                break
    else:
        default_model = next(
            (m for k, _, m in PROVIDER_CHOICES if k == provider_name),
            "local-model",
        )
        provider_label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
        console.print(f"  [dim]Default for {provider_label}:[/dim] [bold]{default_model}[/bold]")
        console.print()
        while True:
            model_choice = Prompt.ask(
                "  [bold]Model name[/bold] [dim](b=back)[/dim]",
                default=default_model,
                console=console,
            ).strip()
            if model_choice.lower() in ("b", "back"):
                raise GoBack()
            model = model_choice or default_model
            if _is_known_litellm_model(model):
                break
            console.print(f"  [yellow]Model '{model}' not found.[/yellow] Are you sure it's correct?")
            confirm = Prompt.ask("  ", choices=["continue", "retry"], default="retry", console=console)
            if confirm == "continue":
                break

    config["model"] = model
    return config


def setup_safety(config: dict) -> dict:
    console.print()
    console.print(section_rule("Command Safety", step=4, total=TOTAL_STEPS))
    console.print()

    choices = [
        ("risky", "Confirm risky commands only", "Detects destructive, privileged, and network commands"),
        ("all", "Confirm all commands", "Every shell command requires your approval"),
        ("none", "No confirmation", "Commands run without prompts (power user)"),
    ]
    for i, (key, label, desc) in enumerate(choices, 1):
        default_tag = " [green]← default[/green]" if key == "risky" else ""
        console.print(f"  [bold cyan]{i}.[/bold cyan] [bold]{label}[/bold] [dim]— {desc}[/dim]{default_tag}")
    console.print()

    while True:
        choice = Prompt.ask(
            "  [bold]Pick a safety level[/bold] [dim](number or name, b=back)[/dim]",
            default="1",
            console=console,
        ).strip()
        if choice.lower() in ("b", "back"):
            raise GoBack()
        resolved = _resolve_safety(choice, choices)
        if resolved is not None:
            break
        console.print("  [red]Invalid choice. Please pick again.[/red]")

    config["command_safety"] = resolved
    return config


def _resolve_safety(choice: str, choices: list) -> str | None:
    try:
        idx = int(choice)
        if 1 <= idx <= len(choices):
            return choices[idx - 1][0]
        return None
    except ValueError:
        pass
    for key, label, _ in choices:
        if choice.lower() in (key.lower(), label.lower()):
            return key
    return None


def setup_base_url(config: dict) -> dict:
    provider_name = config.get("provider", "openai")

    if provider_name not in LOCAL_PROVIDERS:
        return config

    console.print()
    console.print(section_rule("Base URL", step=5, total=TOTAL_STEPS))

    info = PROVIDERS.get(provider_name, {})
    default_url = info.get("base_url") or ""

    if provider_name == "ollama":
        default_url = default_url or "http://localhost:11434"
    elif provider_name == "lm_studio":
        default_url = default_url or "http://localhost:1234/v1"
    elif provider_name == "vllm":
        default_url = default_url or "http://localhost:8000/v1"

    current = config.get("base_url", "")
    display_default = current or default_url

    console.print()
    console.print(f"  [dim]Default:[/dim] [bold]{display_default}[/bold]")
    console.print()
    url = Prompt.ask(
        "  [bold]Base URL[/bold] [dim](b=back)[/dim]",
        default=display_default,
        console=console,
    ).strip()

    if url.lower() in ("b", "back"):
        raise GoBack()

    config["base_url"] = url or display_default
    return config


def test_connection(config: dict) -> bool:
    from .provider import resolve_api_key, create_client, get_backend

    provider_name = config.get("provider", "openai")
    needs_key = provider_name not in LOCAL_PROVIDERS
    has_key = bool(resolve_api_key(config)) if needs_key else True

    if needs_key and not has_key:
        return False

    console.print()

    try:
        backend = get_backend(config)

        with console.status("  [dim]Testing connection...[/dim]", spinner="dots"):
            if provider_name in LOCAL_PROVIDERS:
                import urllib.request
                import urllib.error
                base_url = config.get("base_url", "")
                if not base_url:
                    info = PROVIDERS.get(provider_name, {})
                    base_url = info.get("base_url", "http://localhost:11434")
                health_url = base_url.rstrip("/")
                if "/v1" in health_url:
                    health_url = health_url.rsplit("/v1", 1)[0]
                req = urllib.request.Request(health_url, method="GET")
                urllib.request.urlopen(req, timeout=5)

            elif backend in ("responses", "openai_compat"):
                client = create_client(config)
                client.models.list()

            elif backend == "litellm":
                from .litellm_compat import import_litellm

                litellm = import_litellm()
                api_key = resolve_api_key(config)
                model = config.get("model", "")
                prefix = PROVIDERS.get(provider_name, {}).get("litellm_prefix")
                litellm_model = f"{prefix}/{model}" if prefix and "/" not in model else model
                litellm.completion(
                    model=litellm_model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    api_key=api_key,
                )

        console.print("  [green]✓[/green] [bold green]Connected![/bold green]")
        return True

    except Exception as e:
        err = str(e)
        if len(err) > 120:
            err = err[:120] + "..."
        console.print(f"  [yellow]✗[/yellow] [yellow]Connection failed[/yellow]")
        console.print(f"    [dim]{err}[/dim]")
        console.print()
        console.print("  [dim]Settings saved — fix the issue and run[/dim] [bold cyan]jarv /setup[/bold cyan]")
        return False


# ---------------------------------------------------------------------------
# Full wizard
# ---------------------------------------------------------------------------

def run_setup_wizard(step: str | None = None) -> dict | None:
    """Run the interactive setup wizard.

    If *step* is given, run only that step (provider, key, model, base_url).
    Returns updated config on success, None if cancelled/incomplete.
    """
    from .config import load_config, save_config
    from .provider import resolve_api_key

    config = load_config()

    if step is None:
        console.print()
        console.print(jarv_panel(
            Text.from_markup(
                "[bold]Welcome to jarv![/bold]\n\n"
                "Let's get you set up. This will only take a moment."
            ),
            title="setup",
        ))

        wizard_steps = [setup_provider, setup_api_key, setup_model, setup_safety, setup_base_url]
        step_idx = 0
        going_back = False
        while step_idx < len(wizard_steps):
            fn = wizard_steps[step_idx]
            if going_back and fn == setup_api_key and config.get("provider") in LOCAL_PROVIDERS:
                step_idx -= 1
                continue
            going_back = False
            try:
                config = fn(config)
                step_idx += 1
            except GoBack:
                if step_idx > 0:
                    step_idx -= 1
                    going_back = True
                else:
                    console.print("\n[dim]Setup cancelled.[/dim]\n")
                    return None
        save_config(config)
        test_connection(config)
    elif step == "provider":
        config = setup_provider(config)
        save_config(config)
    elif step == "key":
        config = setup_api_key(config)
        save_config(config)
    elif step == "model":
        config = setup_model(config)
        save_config(config)
    elif step == "safety":
        config = setup_safety(config)
        save_config(config)
    elif step == "base_url":
        config = setup_base_url(config)
        save_config(config)
        test_connection(config)
    else:
        console.print(f"  [red]Unknown setup step '{step}'.[/red]")
        console.print(f"  [dim]Available: provider, key, model, safety, base_url[/dim]")
        return config

    # --- Done summary ---
    console.print()
    provider_name = config.get("provider", "openai")
    provider_label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
    model = config.get("model", "")
    needs_key = provider_name not in LOCAL_PROVIDERS
    has_key = bool(resolve_api_key(config)) if needs_key else True

    if has_key:
        safety = config.get("command_safety", "risky")
        safety_labels = {"all": "Confirm all", "risky": "Confirm risky", "none": "No confirmation"}
        audit = config.get("audit", True)
        auto_approve = config.get("auditor_auto_approve", True)
        audit_label = ("On (auto-approve)" if auto_approve else "On (recommend only)") if audit else "Off"
        console.print(jarv_panel(
            Text.from_markup(
                f"[bold green]You're all set![/bold green]\n\n"
                f"  [dim]Provider[/dim]  [bold]{provider_label}[/bold]\n"
                f"  [dim]Model   [/dim]  [bold]{model}[/bold]\n"
                f"  [dim]Safety  [/dim]  [bold]{safety_labels.get(safety, safety)}[/bold]\n"
                f"  [dim]Audit   [/dim]  [bold]{audit_label}[/bold]\n\n"
                f"[dim]Type [bold]jarv[/bold] to start chatting, or [bold]jarv /settings[/bold] to tweak settings.[/dim]"
            ),
            title="ready",
        ))
    else:
        console.print(jarv_panel(
            Text.from_markup(
                f"[bold yellow]Almost there![/bold yellow]\n\n"
                f"  [dim]Provider[/dim]  [bold]{provider_label}[/bold]\n"
                f"  [dim]API key [/dim]  [bold red]missing[/bold red]\n"
                f"  [dim]Model   [/dim]  [bold]{model}[/bold] [green]✓ saved[/green]"
            ),
            title="setup",
        ))
        _show_env_instructions(provider_name)
    console.print()

    return config


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _prompt_api_key(provider_name: str) -> str:
    label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
    key_url = PROVIDERS.get(provider_name, {}).get("key_url", "")
    if key_url:
        console.print(f"  [dim]Get a key at[/dim] [cyan]{key_url}[/cyan]")
        console.print()
    while True:
        key = Prompt.ask(f"  Enter your {label} API key [dim](b=back)[/dim]", console=console).strip()
        if key.lower() in ("b", "back"):
            raise GoBack()
        if len(key) <= 5:
            console.print("  [red]That doesn't look like a valid key.[/red]")
            continue
        pattern = KEY_PATTERNS.get(provider_name)
        if pattern and not re.match(pattern, key):
            expected_prefix = pattern.split(".")[0].replace("^", "").replace("\\", "")
            console.print(f"  [yellow]Key doesn't match expected format for {label} (expected prefix '{expected_prefix}').[/yellow]")
            use_anyway = Prompt.ask("  Use anyway?", choices=["y", "n"], default="n", console=console)
            if use_anyway == "n":
                continue
        return key


def _resolve_provider(choice: str) -> str | None:
    try:
        idx = int(choice)
        if 1 <= idx <= len(PROVIDER_CHOICES):
            return PROVIDER_CHOICES[idx - 1][0]
        return None
    except ValueError:
        pass
    needle = choice.lower().replace(" ", "").replace("_", "")
    for key, label, _ in PROVIDER_CHOICES:
        if needle in (key.lower(), label.lower()):
            return key
    for key, label, _ in PROVIDER_CHOICES:
        key_norm = key.lower().replace("_", "")
        label_norm = label.lower().replace(" ", "").replace("_", "")
        if needle in key_norm or needle in label_norm or key_norm.startswith(needle):
            return key
    return None


def _is_known_litellm_model(model_name: str) -> bool:
    try:
        import json
        from importlib.resources import files
        from .litellm_compat import import_litellm

        import_litellm()
        data = json.loads(
            files("litellm")
            .joinpath("model_prices_and_context_window_backup.json")
            .read_text(encoding="utf-8")
        )
        return model_name in data
    except Exception:
        return False


def _resolve_model(provider_name: str, choice: str) -> str | None:
    models = PROVIDER_MODELS.get(provider_name, [])
    try:
        idx = int(choice)
        if 1 <= idx <= len(models):
            return models[idx - 1][0]
        return None
    except ValueError:
        pass
    for name, _ in models:
        if choice.lower() == name.lower():
            return name
    if _is_known_litellm_model(choice):
        return choice
    return None
