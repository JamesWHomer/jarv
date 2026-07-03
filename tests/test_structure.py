import importlib

from jarv.provider import KEY_PATTERNS, LOCAL_PROVIDERS, PROVIDERS
from jarv.provider_catalog import FALLBACK_PROVIDER_MODELS, PROVIDER_CHOICES
import ast
from pathlib import Path


def test_handler_specs_resolve_to_their_defining_modules():
    from jarv.command_registry import HANDLER_SPECS

    for name, (module_name, attribute) in HANDLER_SPECS.items():
        handler = getattr(importlib.import_module(module_name), attribute)
        assert handler.__module__ == module_name, name


def test_provider_catalog_covers_setup_choices():
    provider_keys = set(PROVIDERS)
    choice_keys = {key for key, _label, _default_model in PROVIDER_CHOICES}

    assert choice_keys <= provider_keys
    assert LOCAL_PROVIDERS <= provider_keys
    assert set(KEY_PATTERNS) <= provider_keys

    for provider, _label, default_model in PROVIDER_CHOICES:
        preset_models = {
            model
            for model, _description in FALLBACK_PROVIDER_MODELS.get(provider, [])
        }
        assert provider in LOCAL_PROVIDERS or default_model in preset_models


def _internal_import_graph() -> dict[str, set[str]]:
    modules = {path.stem: path for path in Path("jarv").glob("*.py")}
    graph = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                target = node.module.split(".")[0]
                if target in modules:
                    graph[name].add(target)
            elif isinstance(node, ast.ImportFrom) and node.level == 1:
                for alias in node.names:
                    if alias.name in modules:
                        graph[name].add(alias.name)
    return graph


def _strong_components(graph: dict[str, set[str]]) -> list[set[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in graph[node]:
            if target not in indexes:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[target])

        if lowlinks[node] == indexes[node]:
            component = set()
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.add(target)
                if target == node:
                    break
            components.append(component)

    for node in graph:
        if node not in indexes:
            visit(node)
    return components


def test_cleanup_import_cycles_stay_broken():
    graph = _internal_import_graph()
    components = _strong_components(graph)

    assert {"agent", "context_budget", "orchestrator"} not in components
    assert "agent" not in graph["context_budget"]
    assert "config" not in graph["history"]
    assert "provider" not in graph["config"]
    assert "session_commands" not in graph["session_browser"]
    assert "session_commands" not in graph["session_render"]
    assert "session_commands" not in graph["session_store"]
    assert "settings_interactive" not in graph["settings_command"]
    assert "settings_command" not in graph["settings_refresher"]
