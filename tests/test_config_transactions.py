import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from jarv import commands, config as config_module, settings_command, storage
from jarv.config_schema import build_default_config, validate_config_fields


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", path)
    monkeypatch.setattr("jarv.history.migrate_flat_session_files", lambda: None)
    return path


def test_independent_config_changes_survive_repeated_saves(config_path):
    model_session = config_module.load_config()
    colour_session = config_module.load_config()
    model_session["model"] = "other-model"
    config_module.save_config(model_session)
    colour_session["colour"] = False
    config_module.save_config(colour_session)
    colour_session["headsup_border"] = False
    config_module.save_config(colour_session)

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["model"] == "other-model"
    assert saved["colour"] is False
    assert saved["headsup_border"] is False


def test_nested_config_changes_merge(config_path):
    first = config_module.load_config()
    second = config_module.load_config()
    first["api_keys"]["openai"] = "first-key"
    second["api_keys"]["anthropic"] = "second-key"
    config_module.save_config(first)
    config_module.save_config(second)
    assert config_module.load_config()["api_keys"] == {
        "openai": "first-key", "anthropic": "second-key",
    }


def test_conflicting_config_save_preserves_winner(config_path):
    first = config_module.load_config()
    second = config_module.load_config()
    first["model"] = "first-model"
    second["model"] = "second-model"
    second["colour"] = False
    config_module.save_config(first)
    # A later read must not replace the baseline belonging to the stale editor.
    config_module.load_config()

    with pytest.raises(SystemExit) as error:
        config_module.save_config(second)
    assert error.value.code == 1
    assert isinstance(error.value.__context__, storage.StorageConflict)
    saved = config_module.load_config()
    assert saved["model"] == "first-model"
    assert saved["colour"] is True


def test_settings_edits_keep_snapshot_across_saves(config_path):
    first = config_module.load_config()
    second = config_module.load_config()
    first["model"] = "other-model"
    config_module.save_config(first)
    row = next(row for row in settings_command._settings_rows(second)
               if row["key"] == "colour")

    settings_command._settings_apply_quick(row, second)
    # Reads and saves in the same process must not affect this editor's baseline.
    config_module.load_config()
    settings_command._settings_apply_quick(row, second)
    saved = config_module.load_config()
    assert saved["model"] == "other-model"
    assert saved["colour"] is True


def test_failed_settings_save_preserves_live_snapshot(config_path):
    first = config_module.load_config()
    second = config_module.load_config()
    before = copy.deepcopy(second)
    first["command_timeout"] = 42
    config_module.save_config(first)
    row = next(row for row in settings_command._settings_rows(second)
               if row["key"] == "command_timeout")

    with pytest.raises(SystemExit):
        settings_command._settings_commit_edit({"row": row, "buffer": "99"}, second)
    assert second == before
    assert second.baseline == before.baseline
    assert config_module.load_config()["command_timeout"] == 42


@pytest.mark.parametrize("command,args,key,expected", [
    (commands.cmd_set, ["command_timeout", "42"], "command_timeout", 42),
    (commands.cmd_unset, ["colour"], "colour", True),
    (commands.cmd_unset, ["custom"], "custom", None),
])
def test_setting_commands_preserve_snapshot(config_path, monkeypatch, command, args, key, expected):
    initial = config_module.load_config()
    initial.update(colour=False, custom="remove-me")
    config_module.save_config(initial)
    validate = config_module.validate_config

    def concurrent_edit(candidate):
        other = config_module.load_config()
        other["headsup_border"] = False
        config_module.save_config(other)
        return validate(candidate)

    monkeypatch.setattr(config_module, "validate_config", concurrent_edit)
    assert command(args) == 0
    saved = config_module.load_config()
    assert saved.get(key) == expected
    assert saved["headsup_border"] is False


def test_failed_config_commit_preserves_file_and_baseline(config_path, monkeypatch):
    config = config_module.load_config()
    original_bytes = config_path.read_bytes()
    baseline = copy.deepcopy(config.baseline)
    config["colour"] = False
    replace = storage.os.replace

    def fail(source, destination):
        if Path(destination).name == ".jarv-transaction.json":
            raise OSError("disk full")
        return replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(SystemExit) as error:
        config_module.save_config(config)
    assert error.value.code == 1
    assert config_path.read_bytes() == original_bytes
    assert config.baseline == baseline
    assert not list(config_path.parent.glob(".jarv-*"))


def test_interrupted_config_replacement_recovers_on_load(config_path, monkeypatch):
    config = config_module.load_config()
    original_bytes = config_path.read_bytes()
    config["model"] = "recovered-model"
    replace = storage.os.replace

    def fail(source, destination):
        if Path(destination) == config_path:
            raise OSError("interrupted replacement")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(storage.os, "replace", fail)
        with pytest.raises(SystemExit):
            config_module.save_config(config)
    assert config_path.read_bytes() == original_bytes
    assert (config_path.parent / ".jarv-transaction.json").exists()
    assert config_module.load_config()["model"] == "recovered-model"
    assert not (config_path.parent / ".jarv-transaction.json").exists()
    assert not config_path.with_suffix(".json.bak").exists()


@pytest.mark.parametrize("content", [b"{broken", b"null", b"[]", b"\xff"])
def test_invalid_config_is_reported_without_replacement(config_path, content):
    config_path.write_bytes(content)
    with pytest.raises(SystemExit) as error:
        config_module.load_config()
    assert error.value.code == 1
    assert config_path.read_bytes() == content
    assert not config_path.with_suffix(".json.bak").exists()


@pytest.mark.parametrize("initial", ["missing", "legacy", "current"])
def test_two_processes_merge_config_changes(config_path, initial):
    if initial == "current":
        config_module.load_config()
    elif initial == "legacy":
        config_path.write_text(json.dumps({
            "monochrome": False, "api_key": "legacy-key", "provider": "openai",
        }), encoding="utf-8")
    script = '''
import json, sys
from pathlib import Path
from jarv import config, history
config.CONFIG_FILE = Path(sys.argv[1])
config.CONFIG_DIR = config.CONFIG_FILE.parent
history.migrate_flat_session_files = lambda: None
data = config.load_config()
print("ready", flush=True)
sys.stdin.readline()
data[sys.argv[2]] = json.loads(sys.argv[3])
config.save_config(data)
'''
    workers = [subprocess.Popen(
        [sys.executable, "-c", script, str(config_path), key, json.dumps(value)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    ) for key, value in (("model", "other-model"), ("colour", False))]
    try:
        for worker in workers:
            assert worker.stdout.readline().strip() == "ready"
        for worker in workers:
            worker.stdin.write("go\n")
            worker.stdin.flush()
        for worker in workers:
            _, error = worker.communicate(timeout=20)
            assert worker.returncode == 0, error
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait()
    saved = config_module.load_config()
    assert saved["model"] == "other-model"
    assert saved["colour"] is False
    if initial == "legacy":
        assert saved["api_keys"] == {"openai": "legacy-key"}
        assert saved["api_key"] == ""
        assert "monochrome" not in saved


@pytest.mark.parametrize("key,value", [
    ("interactive_commands", "false"), ("audit", 1), ("colour", None),
    ("api_keys", None), ("api_keys", []), ("api_keys", {"openai": 123}),
    ("command_timeout", True), ("command_timeout", 1.5),
])
def test_schema_rejects_malformed_types(key, value):
    config = build_default_config()
    config[key] = value
    errors = []
    assert not validate_config_fields(config, report=errors.append)
    assert any(key in error for error in errors)


@pytest.mark.parametrize("existing", [False, True])
def test_loaded_defaults_do_not_share_mutable_state(tmp_path, monkeypatch, existing):
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("jarv.history.migrate_flat_session_files", lambda: None)
    if existing:
        (tmp_path / "config.json").write_text(json.dumps({"model": "gpt-5.4-mini"}))
    before = copy.deepcopy(config_module.DEFAULT_CONFIG)
    loaded = config_module.load_config()
    loaded["api_keys"]["example"] = "secret"
    loaded["disabled_tools"].append("read")
    assert config_module.DEFAULT_CONFIG == before
    assert build_default_config()["api_keys"] == {}


def test_failed_setting_edit_preserves_live_config(monkeypatch):
    config = build_default_config()
    before = copy.deepcopy(config)
    saved = []
    monkeypatch.setattr(settings_command, "save_config", saved.append)
    row = next(row for row in settings_command._settings_rows(config)
               if row["key"] == "tool_output_display_lines")
    returned, _, _, done = settings_command._settings_commit_edit(
        {"row": row, "buffer": "invalid"}, config)
    assert not done
    assert returned is config
    assert config == before
    assert not saved


def test_failed_nested_setting_edit_preserves_live_config(monkeypatch):
    config = build_default_config()
    config["api_keys"] = {"openai": "old-secret"}
    before = copy.deepcopy(config)
    monkeypatch.setattr(settings_command, "validate_config", lambda candidate: False)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "api_key")
    returned, _, _, done = settings_command._settings_commit_edit(
        {"row": row, "buffer": "clear"}, config)
    assert not done
    assert returned is config
    assert config == before


def test_settings_allows_zero_subagent_depth(monkeypatch):
    config = build_default_config()
    saved = []
    monkeypatch.setattr(settings_command, "save_config", saved.append)
    row = next(row for row in settings_command._settings_rows(config)
               if row["key"] == "max_subagent_depth")
    returned, _, _, done = settings_command._settings_commit_edit(
        {"row": row, "buffer": "0"}, config)
    assert done
    assert returned is config
    assert config["max_subagent_depth"] == 0
    assert saved[0]["max_subagent_depth"] == 0
