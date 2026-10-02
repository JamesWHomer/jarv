"""Exercise the real Windows worker, including recovery and command isolation."""

import concurrent.futures
import ctypes
import gc
import os
import platform
import sys
import threading
import time
import weakref

import pytest

from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.powershell_worker import PowerShellWorker, WorkerUnavailable, _Output
from jarv.shell import MAX_CAPTURE_CHARS, ShellState, execute_command


@pytest.mark.parametrize('chunk_size', [1, 2, 7, 31, 8192])
def test_output_framing_preserves_unicode_partial_lines_and_marker_prefixes(chunk_size):
    fence = '\x1ejarv-test-fence\x1f'
    text = '你好\r\nno newline\x1ejarv-test-other\ré'
    data = (text + fence + 'late output must not leak').encode('utf-8')
    output = _Output(fence)
    for start in range(0, len(data), chunk_size):
        output.feed(data[start:start + chunk_size])
    assert output.done
    assert output.buffer.text() == text.replace('\r\n', '\n')


def test_output_eof_flushes_partial_fence_and_utf8():
    output = _Output('fence')
    output.feed(b'body\r\nfen')
    output.finish()
    assert output.buffer.text() == 'body\nfen'


windows = pytest.mark.skipif(platform.system() != 'Windows', reason='Windows PowerShell worker')


@pytest.fixture
def state():
    value = ShellState.initial()
    yield value
    value.close()


def run(state, command, **kwargs):
    return execute_command(command, shell_state=state, timeout=kwargs.pop('timeout', 15), **kwargs)


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def assert_process_exited(pid):
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    api.OpenProcess.restype = ctypes.c_void_p
    api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = api.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
    if handle:
        try:
            assert api.WaitForSingleObject(handle, 1000) == 0
        finally:
            api.CloseHandle(handle)


@windows
def test_process_reused_but_variables_functions_preferences_and_exit_codes_are_not(state):
    first = run(state, '$global:jarvLeak=42; function global:jarvLeakFn { 42 }; $ErrorActionPreference="Stop"; cmd /c "exit 7"')
    assert first.exit_code == 7
    worker = state._worker
    assert worker is not None and not worker.closed
    result = run(state, "[bool](Get-Variable jarvLeak -ErrorAction SilentlyContinue); [bool](Get-Command jarvLeakFn -ErrorAction SilentlyContinue); $ErrorActionPreference; [bool]$LASTEXITCODE")
    assert result.stdout.splitlines() == ['False', 'False', 'Continue', 'False']
    assert result.exit_code == 0
    assert state._worker is worker


@windows
def test_cwd_env_unicode_and_deletions_survive_without_locking_directory(state, tmp_path):
    directory = tmp_path / "space's [中文]"
    directory.mkdir()
    run(state, f"Set-Location -LiteralPath {quote(directory)}; $env:JARV_WORKER_TEST = \"你好`nsecond\"")
    assert os.path.samefile(state.cwd, directory)
    assert state.env['JARV_WORKER_TEST'] == '你好\nsecond'
    result = run(state, '(Get-Location).Path; $env:JARV_WORKER_TEST; Remove-Item Env:JARV_WORKER_TEST')
    assert str(directory) in result.stdout
    assert '你好\nsecond' in result.stdout
    assert 'JARV_WORKER_TEST' not in state.env
    directory.rmdir()  # An idle worker must not lock the user's cwd.
    assert run(state, "'healed'").stdout.strip() == 'healed'
    assert state.cwd == os.getcwd()


@windows
def test_stdout_stderr_errors_and_no_newline(state):
    result = run(state, '[Console]::Write("out"); [Console]::Error.Write("err")')
    assert (result.stdout, result.stderr, result.exit_code) == ('out', 'err', 0)
    for command, message in [('Write-Error "oops"', 'oops'), ('throw "bad"', 'bad'), ('this-command-does-not-exist-jarv', 'this-command-does-not-exist-jarv')]:
        result = run(state, command)
        assert result.exit_code != 0
        assert message in result.stderr
        assert '#< CLIXML' not in result.stderr
    result = run(state, 'cmd /c "echo native-error 1>&2 & exit 4"')
    assert 'native-error' in result.stderr
    assert result.exit_code == 4
    assert run(state, "'recovered'").exit_code == 0


@windows
def test_preview_arrives_before_command_finishes(state):
    run(state, "'warm'")
    previews = []
    started = time.monotonic()
    result = run(state, "'first'; Start-Sleep -Milliseconds 600; 'last'",
                 on_output=lambda out, err: previews.append((time.monotonic(), out, err)))
    finished = time.monotonic()
    assert result.stdout.splitlines() == ['first', 'last']
    assert any('first' in out and 'last' not in out and finished - at > 0.2
               for at, out, _ in previews)
    assert finished - started >= 0.5


@windows
def test_native_stdin_is_eof_and_large_output_is_drained(state):
    result = run(state, f'& {quote(sys.executable)} -c "import sys; print(repr(sys.stdin.read()))"')
    assert result.stdout.strip() == "''"
    assert result.exit_code == 0
    result = run(state, f"[Console]::Write(('x' * {MAX_CAPTURE_CHARS + 50000})); [Console]::Write('TAIL')")
    assert result.stdout.startswith('x') and result.stdout.endswith('TAIL')
    assert 'command capture limit reached' in result.stdout
    assert len(result.stdout) < MAX_CAPTURE_CHARS + 200
    assert run(state, "'clean'").stdout.strip() == 'clean'


@windows
def test_exit_falls_back_once_and_next_command_recovers(state, tmp_path):
    run(state, "'warm'")
    path = tmp_path / 'executions.txt'
    result = run(state, f"Add-Content -LiteralPath {quote(path)} -Value 'once'; exit 3")
    assert result.exit_code == 3
    assert path.read_text().splitlines() == ['once']
    assert run(state, "'after exit'").stdout.strip() == 'after exit'
    assert not state._worker.closed


@windows
def test_worker_crash_never_replays_a_command_with_side_effects(state, tmp_path):
    path = tmp_path / 'executions.txt'
    result = run(state, f"Add-Content -LiteralPath {quote(path)} -Value 'once'; [Environment]::Exit(7)")
    assert result.exit_code != 0
    assert 'not retried' in result.stderr
    assert path.read_text().splitlines() == ['once']
    assert run(state, "'after crash'").stdout.strip() == 'after crash'


@windows
def test_timeout_preserves_last_committed_state_and_restarts(state):
    run(state, "$env:JARV_WORKER_TEST='old'; 'warm'")
    worker = state._worker
    result = run(state, "$env:JARV_WORKER_TEST='uncommitted'; 'before timeout'; Start-Sleep -Seconds 30", timeout=1)
    assert result.timed_out
    assert 'before timeout' in result.stdout
    assert worker.closed and worker.proc.poll() is not None
    assert state.env['JARV_WORKER_TEST'] == 'old'
    assert run(state, '$env:JARV_WORKER_TEST').stdout.strip() == 'old'


@windows
def test_cancellation_kills_worker_and_native_child_then_recovers(state, tmp_path):
    run(state, "'warm'")
    worker = state._worker
    token = CancellationToken()
    child_seen = []

    def preview(out, err):
        if 'ready' in out:
            child_seen.extend(worker.job._process_ids() - worker.job._host_pids)
            token.cancel()

    # PowerShell escaping for quoted native arguments is version-dependent;
    # use a script file to exercise an actual blocking child unambiguously.
    script = tmp_path / 'child.py'
    script.write_text("import time\nprint('ready', flush=True)\ntime.sleep(30)\n")
    command = f'& {quote(sys.executable)} {quote(script)}'
    with pytest.raises(TurnCancelled):
        run(state, command, cancellation_token=token, on_output=preview)
    assert child_seen
    assert worker.closed and worker.proc.poll() is not None
    for pid in child_seen:
        assert_process_exited(pid)
    # The job handle is gone and all descendants were terminated before reuse.
    assert run(state, "'after cancellation'").stdout.strip() == 'after cancellation'


@windows
def test_copied_state_uses_independent_worker_and_env(state):
    run(state, "$env:JARV_WORKER_TEST='parent'")
    child = state.copy()
    try:
        run(child, "$env:JARV_WORKER_TEST='child'")
        assert child._worker.proc.pid != state._worker.proc.pid
        assert run(state, '$env:JARV_WORKER_TEST').stdout.strip() == 'parent'
        assert run(child, '$env:JARV_WORKER_TEST').stdout.strip() == 'child'
    finally:
        child.close()


@windows
def test_opt_out_releases_worker_and_uses_fresh_process(state):
    run(state, "'warm'")
    worker = state._worker
    assert run(state, "'fresh'", persistent_shell=False).stdout.strip() == 'fresh'
    assert worker.closed and worker.proc.poll() is not None
    assert state._worker is None


@windows
def test_startup_failure_falls_back_before_running_command(state, monkeypatch, tmp_path):
    monkeypatch.setattr(PowerShellWorker, 'start', lambda *args: (_ for _ in ()).throw(WorkerUnavailable('unavailable')))
    path = tmp_path / 'executions.txt'
    assert run(state, f"Add-Content -LiteralPath {quote(path)} -Value 'once'").exit_code == 0
    assert path.read_text().splitlines() == ['once']
    assert state._worker_disabled


@windows
def test_collecting_state_closes_worker():
    state = ShellState.initial()
    run(state, "'warm'")
    worker = state._worker
    reference = weakref.ref(state)
    del state
    gc.collect()
    assert reference() is None
    assert worker.closed and worker.proc.poll() is not None


@windows
def test_concurrent_commands_on_same_state_are_serialized(state):
    run(state, "'warm'")
    def command(index):
        return run(state, f"$env:JARV_WORKER_TEST='{index}'; Start-Sleep -Milliseconds 50; $env:JARV_WORKER_TEST").stdout.strip()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(command, [1, 2])) == ['1', '2']


@windows
def test_cancellation_during_startup_does_not_run_command(state, tmp_path):
    token = CancellationToken()
    timer = threading.Timer(0.05, token.cancel)
    path = tmp_path / 'should-not-exist'
    timer.start()
    try:
        with pytest.raises(TurnCancelled):
            run(state, f"Set-Content -LiteralPath {quote(path)} -Value bad", cancellation_token=token)
    finally:
        timer.cancel()
    assert not path.exists()
    assert state._worker.closed
    if state._worker.proc is not None:
        assert state._worker.proc.poll() is not None
    assert run(state, "'after startup cancellation'").stdout.strip() == 'after startup cancellation'


@windows
def test_unexpected_background_child_retires_worker_and_is_killed(state, tmp_path):
    run(state, "'warm'")
    worker = state._worker
    script = tmp_path / 'launcher.py'
    script.write_text("import subprocess,sys\np=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nprint(p.pid)\n")
    result = run(state, f'& {quote(sys.executable)} {quote(script)}')
    assert result.exit_code == 0
    assert worker.closed
    assert_process_exited(int(result.stdout.strip()))
    assert run(state, "'clean next command'").stdout.strip() == 'clean next command'


@windows
def test_explicit_background_and_process_global_commands_use_fresh_runner(state, monkeypatch):
    from jarv import shell
    from jarv.shell import CommandResult
    commands = []
    def fresh(command, *args):
        commands.append(command)
        return CommandResult(command, '', '', 0)
    monkeypatch.setattr(shell, '_execute_command_fresh', fresh)
    for command in ('Start-Process something', 'saps something', 'Add-Type -TypeDefinition code', 'return', 'exit 2'):
        assert run(state, command).exit_code == 0
        assert state._worker is None
    assert len(commands) == 5


@windows
def test_script_exit_code_and_noninteractive_prompt(state, tmp_path):
    script = tmp_path / 'exit.ps1'
    script.write_text("'script output'; exit 7")
    result = run(state, f'& {quote(script)}')
    assert result.exit_code == 7 and result.stdout.strip() == 'script output'
    result = run(state, 'Read-Host "question"')
    assert result.exit_code == 1 and not result.timed_out
    assert 'NonInteractive' in result.stderr
    assert run(state, "'after prompt'").stdout.strip() == 'after prompt'


@windows
def test_native_stderr_streams_before_exit(state, tmp_path):
    run(state, "'warm'")
    script = tmp_path / 'stderr.py'
    script.write_text("import sys,time\nprint('early error',file=sys.stderr,flush=True)\ntime.sleep(0.6)\nprint('finished')\n")
    previews = []
    result = run(state, f'& {quote(sys.executable)} {quote(script)}',
                 on_output=lambda out, err: previews.append((out, err)))
    assert 'early error' in result.stderr
    assert 'finished' in result.stdout
    assert any('early error' in err and 'finished' not in out for out, err in previews)
