"""Fullscreen /setup wizard.

A single cohesive full-screen experience built on the shared
:class:`~jarv.tui_app.AltScreenApp` loop and the same per-setting submenus that
``/settings`` uses (provider, api_key, model, base_url). Three phases:

* **welcome** — the animated jarv brand mark (shared with heads-up's idle intro)
  greets the user; Enter begins.
* **step** — one shared editor panel per setting. ``←``/``→`` move between steps,
  ``Enter`` commits and advances, ``Esc`` backs out (or cancels on the first).
* **ready** — a calm summary with a background connection check and the commands
  to get going.

The edit rendering and key dispatch are reused verbatim from
:mod:`jarv.settings_editor`; this module owns only the phase sequencing and the
welcome/ready chrome, so there is no inline "Welcome"/"Connected" console output
around the screen any more.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from .config import save_config
from .display import console, terminal_size
from .intro_animation import render_intro
from .provider import LOCAL_PROVIDERS, PROVIDERS
from .settings_command import _settings_begin_edit, _settings_rows
from .settings_editor import apply_catalog_refresh, apply_editor_key, render_editor_panel
from .settings_refresher import _ModelCatalogRefresher
from .tui_app import AltScreenApp
from .tui_frame import panel_width
from .tui_layout import clip_text


# CLI step name -> settings row key
_SETUP_STEP_KEYS = {
    "provider": "provider",
    "key": "api_key",
    "model": "model",
    "base_url": "base_url",
}

_SPIN_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_REPEATABLE_KEYS = frozenset(
    {"UP", "DOWN", "LEFT", "RIGHT", "HOME", "END", "PAGEUP", "PAGEDOWN"}
)


def _setup_step_rows(config: dict) -> list[dict]:
    """Dynamic wizard steps: provider, api_key (cloud only), model, base_url (local only)."""
    rows_by_key = {row["key"]: row for row in _settings_rows(config)}
    provider = config.get("provider", "openai")
    is_local = provider in LOCAL_PROVIDERS
    keys = ["provider"]
    if not is_local:
        keys.append("api_key")
    keys.append("model")
    if is_local:
        keys.append("base_url")
    return [rows_by_key[key] for key in keys if key in rows_by_key]


def _editor_wants_horizontal(edit: dict | None) -> bool:
    """Whether the active editor needs ``←``/``→`` itself (model-not-found prompt).

    That transient picker maps left/right onto its action choices; everywhere
    else the wizard claims left/right for step navigation.
    """
    return bool(edit and edit.get("model_validation_warning"))


# ---------------------------------------------------------------------------
# Connection check + env-var instructions (shown on the ready screen)
# ---------------------------------------------------------------------------

def _detect_shell_and_profile() -> tuple[str, str, str]:
    import os
    import sys

    if sys.platform == "win32":
        return ("PowerShell", 'setx {env_key} "your-key-here"', "$PROFILE")
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return ("zsh", 'export {env_key}="your-key-here"', "~/.zshrc")
    if "fish" in shell:
        return ("fish", 'set -Ux {env_key} "your-key-here"', "~/.config/fish/config.fish")
    return ("bash", 'export {env_key}="your-key-here"', "~/.bashrc")


def _env_instruction_lines(provider_name: str) -> list[Text]:
    """A couple of compact "where to put your key" lines for the ready screen."""
    info = PROVIDERS.get(provider_name, {})
    env_key = info.get("env_key", "API_KEY")
    key_url = info.get("key_url", "")
    _shell, export_template, profile_path = _detect_shell_and_profile()
    export_cmd = export_template.format(env_key=env_key)

    lines: list[Text] = []
    if key_url:
        line = Text()
        line.append("key  ", style="dim")
        line.append(key_url, style="cyan")
        lines.append(line)
    line = Text()
    line.append("set  ", style="dim")
    line.append(export_cmd, style="bold green")
    lines.append(line)
    line = Text()
    line.append(f"     in {profile_path}, then rerun ", style="dim")
    line.append("jarv /setup", style="bold cyan")
    lines.append(line)
    return lines


def probe_connection(config: dict) -> tuple[bool, str | None]:
    """Best-effort reachability check for the configured provider.

    Returns ``(ok, error)`` where ``error`` is ``None`` on success and a short
    message otherwise. Never raises; safe to call from a worker thread.
    """
    from .provider import create_client, get_backend

    provider_name = config.get("provider", "openai")
    try:
        backend = get_backend(config)
        if provider_name in LOCAL_PROVIDERS:
            import urllib.request

            base_url = config.get("base_url", "") or PROVIDERS.get(
                provider_name, {}
            ).get("base_url", "http://localhost:11434")
            health_url = base_url.rstrip("/")
            if "/v1" in health_url:
                health_url = health_url.rsplit("/v1", 1)[0]
            urllib.request.urlopen(
                urllib.request.Request(health_url, method="GET"), timeout=5
            )
        elif backend in ("responses", "openai_compat"):
            from .openai_http import list_models

            client = create_client(config)
            try:
                list_models(client)
            finally:
                client.close()
        elif backend == "anthropic":
            from .anthropic_http import build_payload, create_message

            client = create_client(config)
            try:
                create_message(
                    client,
                    build_payload(
                        config,
                        config.get("model", ""),
                        "",
                        [],
                        [{"role": "user", "content": "hi"}],
                        max_tokens=1,
                    ),
                    max_retries=0,
                )
            finally:
                client.close()
        elif backend == "gemini":
            from .gemini_http import list_models

            client = create_client(config)
            try:
                list_models(client)
            finally:
                client.close()
        return True, None
    except Exception as exc:
        err = str(exc)
        if len(err) > 120:
            err = err[:120] + "..."
        return False, err


class SetupApp(AltScreenApp):
    """The /setup wizard on the single-threaded alt-screen loop."""

    use_mouse_capture = False
    use_bracketed_paste = True
    clear_on_resize = False
    first_paint_label = "setup"

    def __init__(self, config: dict, *, step: str | None = None, render_console=console):
        super().__init__(console=render_console, repeatable_keys=_REPEATABLE_KEYS)
        self.config = config
        self.single_step = step is not None
        self._step_name = step
        self.lock = threading.RLock()

        self.catalog_refresher = _ModelCatalogRefresher()
        self.catalog = SimpleNamespace(
            request=self._request_catalog_refresh,
            cancel_pending=self.catalog_refresher.cancel_pending,
        )

        self.steps = self._initial_steps()
        self.step_idx = 0
        self.edit: dict | None = None
        self._brand_started_at = time.perf_counter()
        self._last_anim_frame = 0.0
        self.conn_status = "idle"  # idle | checking | ok | fail | nokey
        self.conn_error = ""

        if self.single_step:
            self.phase = "step"
            self.edit = self._begin_step(0)
        else:
            self.phase = "welcome"

    # ------------------------------------------------------------------ #
    # Step plumbing
    # ------------------------------------------------------------------ #
    def _initial_steps(self) -> list[dict]:
        if self.single_step:
            wanted = _SETUP_STEP_KEYS.get(self._step_name, self._step_name)
            rows_by_key = {row["key"]: row for row in _settings_rows(self.config)}
            return [rows_by_key[wanted]]
        return _setup_step_rows(self.config)

    def _begin_step(self, idx: int) -> dict:
        row = self.steps[idx]
        edit = _settings_begin_edit(row, self.config)
        if row["key"] in {"model", "auditor_model"}:
            self._request_catalog_refresh(
                str(self.config.get("provider", "openai")),
                target_edit=edit,
                delay=0.01,
            )
        return edit

    def _request_catalog_refresh(self, provider, *, target_edit=None, delay=0) -> None:
        probe = dict(self.config)
        probe["provider"] = provider
        generation = self.catalog_refresher.request(
            probe, self._catalog_refreshed, delay=delay
        )
        if target_edit is not None:
            target_edit["catalog_generation"] = generation

    def _catalog_refreshed(self, provider, choices, generation) -> None:
        with self.lock:
            apply_catalog_refresh(self.edit, self.config, provider, choices, generation)
            if provider == self.config.get("provider"):
                from .reasoning import reconcile_reasoning_effort

                if reconcile_reasoning_effort(self.config) is not None:
                    save_config(self.config)
        self.invalidate()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def on_stop(self) -> None:
        self.catalog_refresher.close()

    def on_interrupt(self) -> None:
        self._finish(None)

    def on_tick(self) -> None:
        # Keep the brand mark (and the ready-screen connection spinner) breathing.
        # Gate to the animation cadence: the loop now wakes far more often than
        # that for input, so repaint at a steady frame rate rather than every wake.
        if self.phase in ("welcome", "ready"):
            now = time.perf_counter()
            if now - self._last_anim_frame >= self.frame_interval:
                self._last_anim_frame = now
                self.invalidate()

    # ------------------------------------------------------------------ #
    # Key handling
    # ------------------------------------------------------------------ #
    def on_key(self, key: str, repeat: int) -> None:
        if self.phase == "welcome":
            self._on_welcome_key(key)
        elif self.phase == "ready":
            self._on_ready_key(key)
        else:
            self._on_step_key(key, repeat)

    def _on_welcome_key(self, key: str) -> None:
        if key in ("ENTER", "RIGHT"):
            self._enter_step_phase()
        elif key == "ESC":
            self._finish(None)

    def _enter_step_phase(self) -> None:
        self.phase = "step"
        self.steps = _setup_step_rows(self.config)
        self.step_idx = 0
        self.edit = self._begin_step(0)

    def _on_step_key(self, key: str, repeat: int) -> None:
        if not _editor_wants_horizontal(self.edit):
            if key == "LEFT":
                self._go_to_prev_step()
                return
            if key == "RIGHT":
                key = "ENTER"

        with self.lock:
            term_w, _term_h = terminal_size(console=self.console)
            self.config, outcome = apply_editor_key(
                self.edit,
                self.config,
                key,
                repeat,
                catalog=self.catalog,
                inner_width=max(1, term_w - 4),
            )
        if outcome.kind == "committed":
            self._advance_step()
        elif outcome.kind == "cancelled":
            if self.single_step or self.step_idx == 0:
                self._finish(None)
            else:
                self._go_to_prev_step()

    def _advance_step(self) -> None:
        if self.single_step:
            self._finish(self.config)
            return
        self.step_idx += 1
        self.steps = _setup_step_rows(self.config)
        if self.step_idx >= len(self.steps):
            self._enter_ready()
        else:
            self.edit = self._begin_step(self.step_idx)

    def _go_to_prev_step(self) -> None:
        if self.single_step or self.step_idx == 0:
            return
        self.step_idx -= 1
        self.steps = _setup_step_rows(self.config)
        self.step_idx = min(self.step_idx, len(self.steps) - 1)
        self.edit = self._begin_step(self.step_idx)

    def _on_ready_key(self, key: str) -> None:
        if key in ("ENTER", "ESC"):
            self._finish(self.config)
        elif key == "LEFT" and not self.single_step:
            self.catalog_refresher.cancel_pending()
            self.phase = "step"
            self.steps = _setup_step_rows(self.config)
            self.step_idx = max(0, len(self.steps) - 1)
            self.edit = self._begin_step(self.step_idx)

    def _finish(self, result: dict | None) -> None:
        self.result = result
        self.stop()

    # ------------------------------------------------------------------ #
    # Ready-phase connection probe
    # ------------------------------------------------------------------ #
    def _enter_ready(self) -> None:
        self.phase = "ready"
        self.result = self.config
        self._brand_started_at = time.perf_counter()
        self._start_connection_probe()

    def _has_key(self) -> bool:
        from .provider import resolve_api_key

        return bool(resolve_api_key(self.config))

    def _start_connection_probe(self) -> None:
        provider = self.config.get("provider", "openai")
        needs_key = provider not in LOCAL_PROVIDERS
        if needs_key and not self._has_key():
            self.conn_status = "nokey"
            return
        self.conn_status = "checking"
        if self.live is None:
            # No event loop is running (unit tests drive on_key directly): resolve
            # synchronously so the ready screen lands on a final state.
            self._apply_probe_result(*probe_connection(self.config))
            return
        threading.Thread(
            target=self._probe_worker, name="setup-conn-probe", daemon=True
        ).start()

    def _probe_worker(self) -> None:
        self._apply_probe_result(*probe_connection(self.config))

    def _apply_probe_result(self, ok: bool, error: str | None) -> None:
        with self.lock:
            self.conn_status = "ok" if ok else "fail"
            self.conn_error = error or ""
        self.invalidate()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self) -> RenderableType:
        term_w, term_h = terminal_size(console=self.console)
        term_h = max(3, term_h)
        width = panel_width(term_w)
        if self.phase == "welcome":
            return self._render_welcome(width, term_h)
        if self.phase == "ready":
            return self._render_ready(width, term_h)
        return self._render_step(width, term_h)

    def _render_step(self, width: int, term_h: int) -> Panel:
        with self.lock:
            edit = self.edit
            label = edit["row"]["label"] if edit is not None else "setup"
            title = (
                label
                if self.single_step
                else f"setup · Step {self.step_idx + 1}/{len(self.steps)} · {label}"
            )
            return render_editor_panel(
                edit,
                self.config,
                panel_width=width,
                height=term_h,
                title=title,
                controls=self._step_controls(),
            )

    def _step_controls(self) -> str:
        if self.single_step:
            return "Enter save   Esc cancel"
        forward = "Enter finish" if self.step_idx + 1 >= len(self.steps) else "→/Enter next"
        back = "Esc cancel" if self.step_idx == 0 else "←/Esc back"
        return f"{forward}   {back}"

    def _render_welcome(self, width: int, term_h: int) -> Panel:
        inner_width = max(1, width - 4)
        inner_height = max(1, term_h - 2)
        elapsed = time.perf_counter() - self._brand_started_at
        rows = render_intro(
            inner_width, inner_height, elapsed, hint=self._welcome_hint(inner_width)
        )
        if not rows:
            rows = self._centered_fallback(
                inner_width,
                inner_height,
                Text("J A R V", style="bold cyan"),
                Text("press enter to begin", style="dim"),
            )
        return self._frame(rows, "setup", "configure jarv", width, term_h)

    @staticmethod
    def _welcome_hint(inner_width: int) -> str:
        if inner_width >= 44:
            return "press enter to begin   ·   esc to cancel"
        if inner_width >= 24:
            return "enter begin · esc cancel"
        return "enter begin"

    def _render_ready(self, width: int, term_h: int) -> Panel:
        inner_width = max(1, width - 4)
        inner_height = max(1, term_h - 2)

        summary = self._center_block(self._ready_summary_lines(), inner_width)
        footer = self._center(
            Text(self._ready_footer(), style="dim italic"), inner_width
        )

        # Crown the summary with the animated brand band when there's room.
        band_height = inner_height - len(summary) - 3
        band = None
        if band_height >= 6:
            band = render_intro(
                inner_width,
                band_height,
                time.perf_counter() - self._brand_started_at,
                hint="",
            )

        content: list[Text] = []
        if band:
            content.extend(band)
            content.append(Text(""))
        content.extend(summary)

        area = max(1, inner_height - 1)  # last row holds the footer
        if not band and len(content) < area:
            top = (area - len(content)) // 2
            content = [Text("") for _ in range(top)] + content
        while len(content) < area:
            content.append(Text(""))
        content = content[:area]
        content.append(footer)
        return self._frame(content, "ready", None, width, term_h)

    def _ready_summary_lines(self) -> list[Text]:
        provider = self.config.get("provider", "openai")
        provider_label = PROVIDERS.get(provider, {}).get("label", provider)
        model = str(self.config.get("model", ""))
        needs_key = provider not in LOCAL_PROVIDERS
        has_key = (not needs_key) or self._has_key()

        lines: list[Text] = []
        if has_key:
            lines.append(Text("You're all set!", style="bold green"))
        else:
            lines.append(Text("Almost there!", style="bold yellow"))
        lines.append(Text(""))

        def field(label: str, value: str, value_style: str = "bold") -> Text:
            line = Text()
            line.append(f"{label:<10}", style="dim")
            line.append(value, style=value_style)
            return line

        lines.append(field("Provider", provider_label))
        if not has_key:
            lines.append(field("API key", "missing", "bold red"))
            lines.append(field("Model", f"{model}  (saved)"))
        else:
            lines.append(field("Model", model))
            safety = self.config.get("command_safety", "risky")
            safety_labels = {
                "all": "Confirm all",
                "risky": "Confirm risky",
                "none": "No confirmation",
            }
            lines.append(field("Safety", safety_labels.get(safety, str(safety))))
            audit = self.config.get("audit", True)
            auto = self.config.get("auditor_auto_approve", True)
            audit_label = (
                ("On (auto-approve)" if auto else "On (recommend only)") if audit else "Off"
            )
            lines.append(field("Audit", audit_label))

        lines.append(Text(""))
        lines.extend(self._ready_status_lines(provider, has_key))
        lines.append(Text(""))
        if has_key:
            lines.append(Text("Type jarv to start chatting.", style="white"))
        tip = Text()
        tip.append("Run ", style="dim")
        tip.append("jarv /help", style="bold cyan")
        tip.append(" for help, or ", style="dim")
        tip.append("jarv /settings", style="bold cyan")
        tip.append(" to tweak settings.", style="dim")
        lines.append(tip)
        return lines

    def _ready_status_lines(self, provider: str, has_key: bool) -> list[Text]:
        if not has_key:
            head = Text("Add your API key to finish:", style="yellow")
            return [head, *_env_instruction_lines(provider)]
        status = self.conn_status
        if status == "ok":
            return [Text("✓ connection verified", style="green")]
        if status == "fail":
            return [Text("⚠ couldn't reach the provider — settings saved", style="yellow")]
        if status == "checking":
            frame = _SPIN_FRAMES[int(time.perf_counter() * 10) % len(_SPIN_FRAMES)]
            return [Text(f"{frame} verifying connection…", style="cyan")]
        return []

    def _ready_footer(self) -> str:
        if self.single_step:
            return "Enter finish"
        return "Enter finish   ← edit answers"

    # ------------------------------------------------------------------ #
    # Small layout helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _center(text: Text, width: int) -> Text:
        pad = max(0, (width - text.cell_len) // 2)
        out = Text(" " * pad, no_wrap=True, overflow="crop")
        out.append_text(text)
        return out

    def _center_block(self, lines: list[Text], width: int) -> list[Text]:
        """Left-pad every line by the same amount so columns align as one block."""
        if not lines:
            return []
        max_w = max(line.cell_len for line in lines)
        pad = max(0, (width - max_w) // 2)
        prefix = " " * pad
        out: list[Text] = []
        for line in lines:
            padded = Text(prefix, no_wrap=True, overflow="crop")
            padded.append_text(line)
            out.append(padded)
        return out

    def _centered_fallback(
        self, inner_width: int, inner_height: int, *texts: Text
    ) -> list[Text]:
        block = [self._center(text, inner_width) for text in texts]
        top = max(0, (inner_height - len(block)) // 2)
        rows = [Text("") for _ in range(top)] + block
        while len(rows) < inner_height:
            rows.append(Text(""))
        return rows[:inner_height]

    def _frame(
        self,
        rows: list[RenderableType],
        title: str,
        subtitle: str | None,
        width: int,
        term_h: int,
    ) -> Panel:
        return Panel(
            Group(*rows),
            title=f"[bold bright_white]jarv ▸ {title}[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{clip_text(subtitle, max(1, width - 6))}[/dim]" if subtitle else None,
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=width,
            height=term_h,
        )


def run_setup_interactive(config: dict, *, step: str | None = None) -> dict | None:
    """Run the fullscreen /setup wizard.

    Returns the updated config when setup completes (each committed step is
    persisted immediately by the shared commit path) or ``None`` if cancelled.
    With *step* set, only that single setting is opened.
    """
    if step is not None:
        wanted = _SETUP_STEP_KEYS.get(step, step)
        rows_by_key = {row["key"]: row for row in _settings_rows(config)}
        if wanted not in rows_by_key:
            console.print(f"  [red]Unknown setup step '{step}'.[/red]")
            return None

    app = SetupApp(config, step=step)
    app.run()
    return app.result
