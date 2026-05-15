import os
import sys

from rich.prompt import Prompt
from rich.text import Text

from .display import console, jarv_panel, section_rule
from .provider import PROVIDERS, LOCAL_PROVIDERS


PROVIDER_CHOICES = [
    ("openai", "OpenAI", "gpt-5.4-mini"),
    ("openrouter", "OpenRouter (200+ models)", "anthropic/claude-sonnet-4-20250514"),
    ("anthropic", "Anthropic", "claude-sonnet-4-20250514"),
    ("gemini", "Google Gemini", "gemini-2.5-flash"),
    ("groq", "Groq", "llama-3.3-70b-versatile"),
    ("deepseek", "DeepSeek", "deepseek-chat"),
    ("together", "Together AI", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("fireworks", "Fireworks AI", "accounts/fireworks/models/llama4-scout-instruct-basic"),
    ("ollama", "Ollama (local)", "llama3"),
    ("lm_studio", "LM Studio (local)", "local-model"),
    ("vllm", "vLLM (local)", "local-model"),
]

OPENAI_MODELS = [
    ("gpt-5.5", "Largest, slowest, smartest"),
    ("gpt-5.4-mini", "Smaller, faster, cheaper"),
    ("gpt-5.4-nano", "Smallest, fastest, cheapest"),
]


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


def run_setup_wizard() -> dict | None:
    """Run the interactive setup wizard. Returns updated config or None if the
    user needs to set their env var first."""
    from .config import load_config, save_config
    from .provider import resolve_api_key

    console.print()
    console.print(jarv_panel(
        Text.from_markup(
            "[bold]Welcome to jarv![/bold]\n\n"
            "Let's get you set up. This will only take a moment."
        ),
        title="setup",
    ))

    # --- Provider ---
    console.print()
    console.print(section_rule("Provider"))
    console.print()

    for i, (key, label, _) in enumerate(PROVIDER_CHOICES, 1):
        default_tag = " [bold green](default)[/bold green]" if i == 1 else ""
        console.print(f"  [bold cyan]{i:>2}.[/bold cyan] [bold]{label}[/bold]{default_tag}")
    console.print()

    choice = Prompt.ask(
        "  Pick a provider [dim](number or name, Enter for default)[/dim]",
        default="1",
        console=console,
    ).strip()

    provider_name = _resolve_provider(choice)
    config = load_config()
    config["provider"] = provider_name

    # --- API key ---
    console.print()
    console.print(section_rule("API Key"))

    if provider_name in LOCAL_PROVIDERS:
        console.print(f"\n  [green]No API key needed[/green] for {PROVIDERS[provider_name]['label']}.")
    else:
        config_snapshot = {**config, "provider": provider_name}
        env_key_name = PROVIDERS.get(provider_name, {}).get("env_key", "")
        api_key = resolve_api_key(config_snapshot)
        if api_key:
            masked = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
            source = f"from {env_key_name}" if env_key_name and os.environ.get(env_key_name, "") else "from config"
            console.print(f"\n  [green]Found[/green] API key [dim]({masked}, {source})[/dim]")
        else:
            label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
            console.print(f"\n  [yellow]No API key found[/yellow] for {label}.")
            console.print()
            api_key = _prompt_api_key(provider_name)
            config["api_key"] = api_key

    # --- Model ---
    console.print()
    console.print(section_rule("Model"))
    console.print()

    if provider_name == "openai":
        for i, (name, desc) in enumerate(OPENAI_MODELS, 1):
            default_tag = " [bold green](default)[/bold green]" if i == 1 else ""
            console.print(f"  [bold cyan]{i}.[/bold cyan] [bold]{name}[/bold] — [dim]{desc}[/dim]{default_tag}")
        console.print()
        model_choice = Prompt.ask(
            "  Pick a model [dim](number or name, Enter for default)[/dim]",
            default="1",
            console=console,
        ).strip()
        model = _resolve_openai_model(model_choice)
    else:
        default_model = next(
            (m for k, _, m in PROVIDER_CHOICES if k == provider_name),
            "default",
        )
        provider_label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
        console.print(f"  [dim]Default for {provider_label}:[/dim] [bold]{default_model}[/bold]")
        console.print()
        model_choice = Prompt.ask(
            "  Model name [dim](Enter for default)[/dim]",
            default=default_model,
            console=console,
        ).strip()
        model = model_choice or default_model

    config["model"] = model
    save_config(config)

    # --- Done ---
    console.print()
    provider_label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
    needs_key = provider_name not in LOCAL_PROVIDERS
    has_key = bool(resolve_api_key(config)) if needs_key else True

    if has_key:
        console.print(jarv_panel(
            Text.from_markup(
                f"[bold green]You're all set![/bold green]\n\n"
                f"  Provider  [bold]{provider_label}[/bold]\n"
                f"  Model     [bold]{model}[/bold]\n\n"
                f"[dim]Run [bold]jarv /config[/bold] to view settings or [bold]jarv /set <key> <value>[/bold] to change them.[/dim]"
            ),
            title="ready",
        ))
    else:
        console.print(jarv_panel(
            Text.from_markup(
                f"[bold yellow]Almost there![/bold yellow]\n\n"
                f"  Provider  [bold]{provider_label}[/bold]\n"
                f"  API key   [bold red]missing[/bold red]\n"
                f"  Model     [bold]{model}[/bold] [green]saved[/green]"
            ),
            title="setup",
        ))
    console.print()

    return config


def _prompt_api_key(provider_name: str) -> str:
    label = PROVIDERS.get(provider_name, {}).get("label", provider_name)
    key_url = PROVIDERS.get(provider_name, {}).get("key_url", "")
    if key_url:
        console.print(f"  [dim]Get a key at[/dim] [cyan]{key_url}[/cyan]")
        console.print()
    while True:
        key = Prompt.ask(f"  Enter your {label} API key", console=console).strip()
        if len(key) > 5:
            return key
        console.print("  [red]That doesn't look like a valid key[/red]")


def _resolve_provider(choice: str) -> str:
    try:
        idx = int(choice)
        if 1 <= idx <= len(PROVIDER_CHOICES):
            return PROVIDER_CHOICES[idx - 1][0]
    except ValueError:
        pass
    for key, label, _ in PROVIDER_CHOICES:
        if choice.lower() in (key.lower(), label.lower()):
            return key
    return PROVIDER_CHOICES[0][0]


def _resolve_openai_model(choice: str) -> str:
    try:
        idx = int(choice)
        if 1 <= idx <= len(OPENAI_MODELS):
            return OPENAI_MODELS[idx - 1][0]
    except ValueError:
        pass
    for name, _ in OPENAI_MODELS:
        if choice.lower() == name.lower():
            return name
    return OPENAI_MODELS[0][0]
