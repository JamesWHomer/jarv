"""Reusable Windows PowerShell process with isolated command runspaces.

Commands/results travel over a private named pipe; stdin remains DEVNULL.
Random per-command fences on both output pipes drain all synchronous output
before completion. A Windows job owns the entire tree. Once a request is sent
it is never retried, even if the worker dies before acknowledging completion.
"""

import atexit
import base64
import codecs
import io
import json
import os
import queue
import subprocess
import threading
import time
import uuid
import weakref

from .cancellation import TurnCancelled
from .shell import (
    CommandResult, _BoundedOutput, _kill_process_tree,
    _STATE_CWD_FILE_VAR, _STATE_ENV_FILE_VAR,
)
from .windows_job import WindowsJob


# Kept in Python so wheels and standalone builds need no extra data files.
# Fresh runspaces preserve the existing cwd/env-only persistence contract.
_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$idleDirectory = $PSHOME
[Console]::OutputEncoding = $utf8
$pipe = New-Object IO.Pipes.NamedPipeServerStream('__PIPE__', [IO.Pipes.PipeDirection]::In)
$replies = New-Object IO.Pipes.NamedPipeServerStream('__PIPE__-reply', [IO.Pipes.PipeDirection]::Out)
try {
    $pipe.WaitForConnection()
    $replies.WaitForConnection()
    $reader = New-Object IO.StreamReader($pipe, $utf8)
    $writer = New-Object IO.StreamWriter($replies, $utf8)
    $writer.AutoFlush = $true
    $writer.WriteLine('{"ready":true}')
    while ($null -ne ($line = $reader.ReadLine())) {
        $request = ConvertFrom-Json -InputObject $line
        $tokens = $null; $parseErrors = $null
        $ast = [Management.Automation.Language.Parser]::ParseInput($request.command, [ref]$tokens, [ref]$parseErrors)
        # Preserve process-level semantics for flow control, background jobs,
        # and explicitly loaded .NET types. Resolve standard aliases too.
        $freshOnly = $ast.Find({param($a)
            if ($a -is [Management.Automation.Language.ExitStatementAst] -or
                $a -is [Management.Automation.Language.ReturnStatementAst] -or
                $a -is [Management.Automation.Language.BreakStatementAst] -or
                $a -is [Management.Automation.Language.ContinueStatementAst]) { return $true }
            if ($a -is [Management.Automation.Language.CommandAst]) {
                $name = $a.GetCommandName()
                if ($name) { $name = ($name -split '\\')[-1] }
                return $name -in @('Add-Type', 'Start-Process', 'start', 'saps',
                    'Start-Job', 'sajb', 'Start-ThreadJob', 'Register-ObjectEvent',
                    'Register-EngineEvent', 'Register-WmiEvent')
            }
            return $false
        }, $true)
        if ($null -ne $freshOnly) {
            $writer.WriteLine('{"fallback":true}')
            continue
        }
        # Replay state even when an interactive/fresh command changed it.
        foreach ($key in @([Environment]::GetEnvironmentVariables().Keys)) {
            [Environment]::SetEnvironmentVariable([string]$key, $null, 'Process')
        }
        foreach ($property in $request.env.PSObject.Properties) {
            [Environment]::SetEnvironmentVariable($property.Name, [string]$property.Value, 'Process')
        }
        [Environment]::CurrentDirectory = $request.cwd
        [Console]::OutputEncoding = $utf8
        $runspace = [runspacefactory]::CreateRunspace($Host)
        $runspace.Open()
        $runspace.SessionStateProxy.Path.SetLocation([Management.Automation.WildcardPattern]::Escape($request.cwd)) > $null
        $ps = [powershell]::Create()
        $ps.Runspace = $runspace
        # Stream and discard error records instead of accumulating them until
        # Invoke returns. The Python reader retains a bounded copy.
        $ps.Streams.Error.add_DataAdded({param($sender, $eventArgs)
            [Console]::Error.WriteLine($sender[$eventArgs.Index].ToString())
            $sender.Clear()
        })
        $capture = @'

$__jarvOk = $?
$__jarvExit = $LASTEXITCODE
$global:__jarvResult = @{
    code = $(if ($__jarvExit -is [int]) { $__jarvExit } elseif ($__jarvOk) { 0 } else { 1 })
    cwd = (Get-Location).Path
}
'@
        try {
            [void]$ps.AddScript($request.command + "`n" + $capture)
            [void]$ps.AddCommand('Out-Default')
            [void]$ps.Invoke()
            $result = $runspace.SessionStateProxy.GetVariable('__jarvResult')
            if ($null -eq $result) { $result = @{code=1} }
        } catch {
            [Console]::Error.WriteLine($_.ToString())
            $result = @{code=1}
        } finally {
            $ps.Dispose()
            $runspace.Dispose()
        }
        if ($result.ContainsKey('cwd')) {
            $result.env = @{}
            foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) {
                $result.env[[string]$entry.Key] = [string]$entry.Value
            }
        }
        # Do not keep the user's directory locked while the worker is idle.
        [Environment]::CurrentDirectory = $idleDirectory
        # User code may have changed console encoding or writers.
        [Console]::OutputEncoding = $utf8
        $stdout = New-Object IO.StreamWriter([Console]::OpenStandardOutput(), $utf8)
        $stderr = New-Object IO.StreamWriter([Console]::OpenStandardError(), $utf8)
        $stdout.AutoFlush = $true; $stderr.AutoFlush = $true
        [Console]::Out.Flush(); [Console]::Error.Flush()
        $stdout.Write($request.fence); $stdout.Flush()
        $stderr.Write($request.fence); $stderr.Flush()
        [Console]::SetOut($stdout); [Console]::SetError($stderr)
        $writer.WriteLine((ConvertTo-Json -InputObject $result -Compress -Depth 3))
    }
} finally {
    $pipe.Dispose()
    $replies.Dispose()
}
'''


class WorkerUnavailable(Exception):
    """No user command was sent; falling back is safe."""

    def __init__(self, message, *, disable=True):
        super().__init__(message)
        self.disable = disable


class _Output:
    """Bounded output with a fence that may straddle arbitrary pipe reads."""

    def __init__(self, fence):
        self.fence = fence.encode('utf-8')
        self.pending = b''
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.buffer = _BoundedOutput()
        self.done = False
        self.cr = ''

    def feed(self, data):
        if self.done:
            return
        data = self.pending + data
        end = data.find(self.fence)
        if end >= 0:
            self.done = True
            body, self.pending = data[:end], b''
        else:
            keep = 0
            for size in range(min(len(data), len(self.fence) - 1), 0, -1):
                if data.endswith(self.fence[:size]):
                    keep = size
                    break
            body = data[:-keep] if keep else data
            self.pending = data[-keep:] if keep else b''
        self._decode(body, final=self.done)

    def _decode(self, data, *, final=False):
        text = self.cr + self.decoder.decode(data, final=final)
        self.cr = ''
        if text.endswith('\r') and not final:
            text, self.cr = text[:-1], '\r'
        if text:
            self.buffer.append(text.replace('\r\n', '\n'))

    def finish(self):
        if not self.done:
            self._decode(self.pending, final=True)
            self.pending = b''
            self.done = True


class _OutputRouter:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams = None

    def read(self, stream, index):
        try:
            while data := stream.read(8192):
                with self.lock:
                    if self.streams is not None:
                        self.streams[index].feed(data)
        except (OSError, ValueError):
            pass
        finally:
            with self.lock:
                if self.streams is not None:
                    self.streams[index].finish()
            stream.close()

    def snapshot(self):
        with self.lock:
            return tuple(part.buffer.text() for part in self.streams)

    def drained(self):
        with self.lock:
            return all(part.done for part in self.streams)


_workers = weakref.WeakSet()


def _close_workers():
    for worker in list(_workers):
        worker.close()


atexit.register(_close_workers)


def _read_replies(pipe, replies):
    try:
        # Buffer variable-size reads; FileIO.readline() makes a Windows read
        # syscall per byte of the environment snapshot otherwise.
        with io.BufferedReader(pipe) as reader:
            for line in reader:
                replies.put(json.loads(line))
    except (OSError, ValueError):
        pass
    finally:
        replies.put(None)


class PowerShellWorker:
    def __init__(self):
        self.proc = None
        self.job = None
        self.pipe = None
        self.reply_pipe = None
        self.router = _OutputRouter()
        self.replies = queue.Queue()
        self.threads = []
        self.closed = False
        self._close_lock = threading.Lock()
        _workers.add(self)

    def _check(self, deadline, token):
        if token is not None:
            token.throw_if_cancelled()
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired('PowerShell worker', 0)

    def start(self, state, deadline, token):
        name = 'jarv-' + uuid.uuid4().hex
        script = base64.b64encode(_SCRIPT.replace('__PIPE__', name).encode('utf-16le')).decode('ascii')
        try:
            proc = subprocess.Popen(
                ['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive',
                 '-ExecutionPolicy', 'Bypass', '-OutputFormat', 'Text', '-EncodedCommand', script],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0, cwd=os.environ.get('SystemRoot', state.cwd), env=state.env,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            # Cancellation can race with Popen before the process is published.
            with self._close_lock:
                if self.closed:
                    _kill_process_tree(proc)
                    self._check(deadline, token)
                    raise WorkerUnavailable('Worker closed during startup')
                self.proc = proc
                self.job = WindowsJob(proc)
                for index, stream in enumerate((self.proc.stdout, self.proc.stderr)):
                    thread = threading.Thread(target=self.router.read, args=(stream, index), daemon=True)
                    thread.start()
                    self.threads.append(thread)
            startup_deadline = min(deadline, time.monotonic() + 5)
            while True:
                self._check(deadline, token)
                if self.proc.poll() is not None or time.monotonic() >= startup_deadline:
                    raise WorkerUnavailable('PowerShell worker did not start')
                try:
                    pipe = open('\\\\.\\pipe\\' + name, 'wb', buffering=0)
                except OSError:
                    time.sleep(0.02)
                    continue
                with self._close_lock:
                    if self.closed:
                        pipe.close()
                        self._check(deadline, token)
                        raise WorkerUnavailable('Worker closed during startup')
                    self.pipe = pipe
                break
            # A client can open the first pipe before PowerShell has even
            # constructed the second one. Do not reopen the connected request
            # pipe while waiting for the reply endpoint to appear.
            while True:
                self._check(deadline, token)
                if self.proc.poll() is not None or time.monotonic() >= startup_deadline:
                    raise WorkerUnavailable('PowerShell reply pipe did not start')
                try:
                    reply_pipe = open('\\\\.\\pipe\\' + name + '-reply', 'rb', buffering=0)
                except OSError:
                    time.sleep(0.02)
                    continue
                with self._close_lock:
                    if self.closed:
                        reply_pipe.close()
                        self._check(deadline, token)
                        raise WorkerUnavailable('Worker closed during startup')
                    self.reply_pipe = reply_pipe
                    thread = threading.Thread(target=_read_replies, args=(self.reply_pipe, self.replies), daemon=True)
                    thread.start()
                    self.threads.append(thread)
                break
            while True:
                self._check(deadline, token)
                try:
                    reply = self.replies.get(timeout=0.02)
                    if reply != {'ready': True}:
                        raise WorkerUnavailable('Invalid worker greeting')
                    self.job.record_host_processes()
                    break
                except queue.Empty:
                    if time.monotonic() >= startup_deadline:
                        raise WorkerUnavailable('PowerShell worker did not become ready')
        except (OSError, ValueError) as exc:
            raise WorkerUnavailable(str(exc)) from exc

    def run(self, command, state, timeout, token, on_output):
        deadline = time.monotonic() + timeout
        unregister = token.register(self.close) if token else lambda: None
        sent = False
        try:
            self._check(deadline, token)
            if self.proc is None:
                self.start(state, deadline, token)
            self._check(deadline, token)
            fence = '\x1ejarv-' + uuid.uuid4().hex + '\x1f'
            with self.router.lock:
                self.router.streams = (_Output(fence), _Output(fence))
            request = json.dumps({
                'command': command, 'cwd': state.cwd,
                'env': state.env if state.env is not None else dict(os.environ),
                'fence': fence,
            }, ensure_ascii=True).encode('utf-8') + b'\n'
            # Once any bytes might have reached the worker, never replay this
            # command on another process (it may already have side effects).
            sent = True
            view = memoryview(request)
            while view:
                self._check(deadline, token)
                count = self.pipe.write(view)
                if not count:
                    raise OSError('PowerShell control pipe closed')
                view = view[count:]
            reply = None
            next_preview = 0
            while True:
                self._check(deadline, token)
                if reply is None:
                    try:
                        reply = self.replies.get(timeout=0.02)
                        if reply is None:
                            raise OSError('PowerShell worker exited before completing the command')
                        if reply.get('fallback') is True:
                            raise WorkerUnavailable('Command requires a fresh process', disable=False)
                    except queue.Empty:
                        pass
                else:
                    time.sleep(0.005)
                if on_output is not None and time.monotonic() >= next_preview:
                    on_output(*self.router.snapshot())
                    next_preview = time.monotonic() + 0.1
                if reply is not None and self.router.drained():
                    break
                if self.proc.poll() is not None:
                    raise OSError('PowerShell worker exited before completing the command')
            if not isinstance(reply.get('code'), int):
                raise OSError('Invalid PowerShell command result')
            if isinstance(reply.get('cwd'), str) and os.path.isdir(reply['cwd']):
                state.cwd = reply['cwd']
            if isinstance(reply.get('env'), dict):
                state.env = {k: v for k, v in reply['env'].items()
                             if isinstance(v, str) and not k.startswith('=')
                             and k.upper() not in (_STATE_CWD_FILE_VAR, _STATE_ENV_FILE_VAR)}
            stdout, stderr = self.router.snapshot()
            if self.job.has_children():
                # Never let an orphan's later output contaminate the next call.
                self.close()
            return CommandResult(command, stdout, stderr, reply['code'], timeout=timeout)
        except WorkerUnavailable:
            self.close()
            raise
        except (KeyboardInterrupt, TurnCancelled):
            self.close()
            raise
        except Exception as exc:
            self.close()
            if token is not None:
                token.throw_if_cancelled()
            stdout, stderr = self.router.snapshot() if self.router.streams else ('', '')
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            if not sent and not timed_out:
                raise WorkerUnavailable(str(exc)) from exc
            if not timed_out:
                stderr += f'\n[PowerShell worker error: {exc}; command was not retried]\n'
            return CommandResult(command, stdout, stderr, None, timed_out=timed_out, timeout=timeout)
        finally:
            unregister()

    def close(self):
        with self._close_lock:
            if self.closed:
                return
            self.closed = True
            if self.job is not None:
                self.job.close()
            elif self.proc is not None:
                _kill_process_tree(self.proc)
            if self.proc is not None:
                try:
                    self.proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(self.proc)
            deadline = time.monotonic() + 1
            for thread in self.threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=max(0, deadline - time.monotonic()))
            for stream in (self.pipe, self.reply_pipe,
                           getattr(self.proc, 'stdout', None), getattr(self.proc, 'stderr', None)):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
