import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from jarv import shell
from jarv.artifacts import ArtifactStore
from jarv.config import DEFAULT_CONFIG
from jarv.edit_tool import _load_file
from jarv.pdf_extract import extract_pdf_text
from jarv.read_tool import _resolve_source, dispatch_read_tool
from jarv.retained_outputs import RetainedOutputStore


def test_capture_keeps_head_tail_and_reports_discarded_middle():
    capture = shell._BoundedOutput(20)
    for _ in range(100):
        capture.append("abcdefghij")
    assert len(capture.head) + capture.tail_size == 20
    assert capture.text().startswith("abcdefghij")
    assert capture.text().endswith("abcdefghij")
    assert "980 characters omitted" in capture.text()
    assert capture.text() is capture.text()  # Unchanged snapshots reuse the string.


def test_interactive_delta_stays_correct_when_capture_rolls_over():
    proc = SimpleNamespace(stdout=None, stderr=None, poll=lambda: None)
    process = shell.InteractiveCommandProcess("cmd", proc)
    process._stdout_parts = shell._BoundedOutput(20)
    process._stdout_parts.append("a" * 25)
    process.snapshot(consume=True)
    process._stdout_parts.append("new")
    assert process.snapshot(consume=True).stdout_delta == "new"
    assert process.snapshot().stdout_delta == ""


def test_posix_kill_escalates_even_when_shell_exits_before_children():
    proc = Mock(pid=123)
    with patch.object(signal, "SIGKILL", 9, create=True), patch.object(shell.platform, "system", return_value="Linux"), patch.object(
        shell.os, "killpg", create=True
    ) as killpg:
        shell._kill_process_tree(proc)
    assert [call.args for call in killpg.call_args_list] == [
        (123, signal.SIGTERM), (123, getattr(signal, "SIGKILL", 9))
    ]
    assert all(call.kwargs.get("timeout") for call in proc.wait.call_args_list)


def test_windows_taskkill_timeout_falls_back_and_reaps():
    proc = Mock(pid=123)
    with patch.object(shell.platform, "system", return_value="Windows"), patch.object(
        shell.subprocess, "run", side_effect=subprocess.TimeoutExpired("taskkill", 1)
    ) as run:
        shell._kill_process_tree(proc)
    assert run.call_args.kwargs["timeout"] > 0
    proc.kill.assert_called_once()
    assert proc.wait.call_args.kwargs["timeout"] > 0


def test_kill_tree_cleans_children_after_parent_has_exited():
    proc = SimpleNamespace(stdout=None, stderr=None, poll=lambda: 0)
    process = shell.InteractiveCommandProcess("cmd", proc)
    with patch.object(shell, "_kill_process_tree") as kill:
        process.kill_tree()
    kill.assert_called_once_with(proc)


def test_windows_taskkill_failure_falls_back_to_direct_kill():
    proc = Mock(pid=123)
    proc.poll.return_value = None
    with patch.object(shell.platform, "system", return_value="Windows"), patch.object(
        shell.subprocess, "run", return_value=SimpleNamespace(returncode=1)
    ):
        shell._kill_process_tree(proc)
    proc.kill.assert_called_once()


def test_local_text_read_retains_only_requested_unicode_page(tmp_path):
    path = tmp_path / "large.txt"
    content = "é中\r\n" * 50_000
    path.write_bytes(content.encode("utf-8"))
    with patch.object(type(path), "read_bytes", side_effect=AssertionError("unbounded read")):
        source = _resolve_source(str(path), visible_labels=set(),
                                 artifact_store=ArtifactStore(),
                                 retained_store=RetainedOutputStore(), config=DEFAULT_CONFIG,
                                 cancellation_token=None, offset=65_533, size=17)
    assert source.content == content[65_533:65_550]
    assert source.total_size == len(content)
    assert len(source.content) == 17


def test_edit_read_is_bounded_before_allocation(tmp_path, monkeypatch):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * 1000)
    monkeypatch.setattr("jarv.edit_tool.MAX_EDIT_FILE_BYTES", 10)
    with path.open("rb") as stream:
        guarded = Mock(wraps=stream)
        context = MagicMock()
        context.__enter__.return_value = guarded
        with patch.object(type(path), "open", return_value=context):
            assert "exceeding" in _load_file(path)
        guarded.read.assert_called_once_with(11)


def test_pdf_stops_extracting_pages_after_text_budget(monkeypatch):
    pages = [Mock() for _ in range(3)]
    for page in pages:
        page.extract_text.return_value = "x" * 100
    reader = SimpleNamespace(is_encrypted=False, pages=pages, metadata=None)
    monkeypatch.setattr("jarv.pdf_extract._load_pdf_reader_class", lambda: lambda *a, **k: reader)
    monkeypatch.setattr("jarv.pdf_extract.MAX_PDF_TEXT_CHARS", 50)
    result = extract_pdf_text(b"%PDF-")
    assert "extraction limit reached" in result.text
    pages[1].extract_text.assert_not_called()
    pages[2].extract_text.assert_not_called()


def test_retention_evicts_oldest_by_total_size_and_explains_expiration(monkeypatch):
    monkeypatch.setattr("jarv.retained_outputs.MAX_RETAINED_TOTAL_CHARS", 30)
    store = RetainedOutputStore()
    first = store.put("a" * 20)
    second = store.put("b" * 20)
    assert store.get(first) is None
    assert store.get(second).content == "b" * 20
    result = dispatch_read_tool({"input": first}, visible_labels=set(),
                               artifact_store=ArtifactStore(), retained_store=store,
                               config=DEFAULT_CONFIG)
    assert "retention limit" in result


def test_noninteractive_process_uses_bounded_reader_and_times_out():
    # A real subprocess exercises both pipes, the timeout and final drain.
    import platform
    command = ("Write-Output before; Start-Sleep -Seconds 5" if platform.system() == "Windows"
               else "printf before; sleep 5")
    result = shell.execute_command(command, timeout=0.8)
    assert result.timed_out
    assert "before" in result.stdout


def test_posix_start_uses_thread_safe_session_creation():
    proc = SimpleNamespace(stdout=None, stderr=None)
    with patch.object(shell.platform, "system", return_value="Linux"), patch.object(
        shell.subprocess, "Popen", return_value=proc
    ) as popen:
        shell.InteractiveCommandProcess.start("echo hi")
    assert popen.call_args.kwargs["start_new_session"] is True
    assert "preexec_fn" not in popen.call_args.kwargs
