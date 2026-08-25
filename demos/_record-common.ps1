# Shared setup for record.ps1 (single/sequential) and record-all.ps1 (parallel).
# Dot-source it (`. "$PSScriptRoot\_record-common.ps1"`): it defines the repo
# paths, the ffmpeg-shim / PATH bootstrap, the recording reasoning effort, and
# the retime step in the caller's scope. See demos/README.md for requirements.
$ErrorActionPreference = 'Stop'

$DemosDir  = $PSScriptRoot
$RepoRoot  = Split-Path $DemosDir -Parent
$OutputDir = "$DemosDir\output"

# VHS bakes the tapes' timing (TypingSpeed 40ms etc.) into the WebP, which plays
# too fast. After recording we rescale every frame delay by this factor
# (>1 = slower) via retime.py — same frames, same file size, only slower.
$PlaybackFactor = 1.2

# High reasoning effort parks demos on a spinner for minutes, so recordings force
# the effort down. (`none` is not an option: most current models reject it, and
# `jarv /set` refuses.) Effort is a global config value, so it is set once for the
# run and the user's own setting is restored afterwards.
$RecordEffort = 'low'

# Put vhs/ttyd/ffmpeg-shim on PATH and compile the shim. Idempotent — safe to
# call once per script run.
function Initialize-RecordEnv {
    # winget's portable installs extend the *user* PATH; a shell opened before
    # the install won't have them yet, so fold both scopes in before looking.
    foreach ($scope in 'Machine', 'User') {
        $env:Path = "$env:Path;" + [Environment]::GetEnvironmentVariable('Path', $scope)
    }
    # ttyd's current Windows builds (the `ttyd.win32.exe` asset, 1.7.5-1.7.7,
    # which is what winget installs) exit the moment a client opens the
    # websocket on Windows 11 26xxx, so VHS waits forever for an xterm canvas
    # that never renders. The old 1.7.2 `ttyd.win10.exe` build works; if it has
    # been fetched into bin\ttyd (see demos/README.md), prefer it over PATH.
    $localTtyd = "$DemosDir\bin\ttyd"
    if (Test-Path "$localTtyd\ttyd.exe") { $env:Path = "$localTtyd;$env:Path" }
    foreach ($bin in 'vhs', 'ttyd', 'ffmpeg', 'jarv', 'python') {
        if (-not (Get-Command $bin -ErrorAction SilentlyContinue)) {
            throw "'$bin' not found on PATH - see the requirements in demos/README.md."
        }
    }

    # Two of VHS's child processes are shimmed (sources in bin\, compiled with
    # the built-in .NET Framework csc.exe):
    #   ffmpeg - VHS hands it no codec options, so .webp output defaults to lossy
    #            VP8 (4:2:0 chroma smears colored text); the shim upgrades .webp
    #            encodes to lossless RGB (VP8L).
    #   ttyd   - the shim drops VHS's `--once`, which otherwise kills the
    #            terminal server mid-setup on Windows (see bin\ttyd-shim.cs).
    # Resolve the real binaries BEFORE the shims shadow them: dropping the shim
    # dir first keeps a second call in the same shell from pointing the shims at
    # themselves.
    $shimDir = "$DemosDir\bin\shim"
    $env:Path = ($env:Path -split ';' | Where-Object { $_ -and $_ -ne $shimDir }) -join ';'
    $env:FFMPEG_SHIM_REAL = (Get-Command ffmpeg).Source
    $env:FFMPEG_SHIM_LOG = "$DemosDir\bin\ffmpeg-shim.log"
    $env:TTYD_SHIM_REAL = (Get-Command ttyd).Source
    foreach ($name in 'ffmpeg', 'ttyd') {
        $shimSrc = "$DemosDir\bin\$name-shim.cs"
        $shimExe = "$shimDir\$name.exe"
        if (-not (Test-Path $shimExe) -or (Get-Item $shimSrc).LastWriteTime -gt (Get-Item $shimExe).LastWriteTime) {
            New-Item -ItemType Directory -Force $shimDir | Out-Null
            & "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe" -nologo -out:$shimExe $shimSrc
            if ($LASTEXITCODE -ne 0) { throw "$name shim compile failed" }
        }
    }
    $env:Path = "$shimDir;$env:Path"

    # Record from the repo root so demos show the real project (cwd in the footer,
    # `cat README.md`, git context). Tapes write to demos/output/ accordingly.
    Set-Location $RepoRoot
    New-Item -ItemType Directory -Force $OutputDir | Out-Null
}

# Every .tape in tapes\ except the shared _common.tape, which holds only the Set
# commands the real tapes Source and is not recordable on its own. Named tapes
# come back in the order asked for, so record.ps1 records them in that order.
function Get-Tapes([string[]]$names) {
    $tapes = Get-ChildItem "$DemosDir\tapes\*.tape" | Where-Object { $_.BaseName -notlike '_*' }
    if (-not $names) { return $tapes }
    foreach ($name in $names) {
        $match = $tapes | Where-Object BaseName -eq $name
        if (-not $match) { throw "No tape named '$name' in tapes\" }
        $match
    }
}

function Set-Effort([string]$effort) {
    $setOutput = jarv /set reasoning_effort $effort
    if ($LASTEXITCODE -ne 0) { throw "jarv /set reasoning_effort $effort failed: $setOutput" }
}

function Get-CurrentEffort {
    (Get-Content "$env:USERPROFILE\.jarv\config.json" | ConvertFrom-Json).reasoning_effort
}

function Restore-Effort($prev) {
    if ($prev) { jarv /set reasoning_effort $prev | Out-Null }
    else { jarv /unset reasoning_effort | Out-Null }
}

# Stash the pristine fast capture (retime.py always scales from .orig so re-runs
# never compound and we can re-time later without re-recording), then rescale the
# frame delays in place.
function Complete-Retime([string[]]$names) {
    $orig = "$OutputDir\.orig"
    New-Item -ItemType Directory -Force $orig | Out-Null
    # @(...) forces an array: a single name would otherwise unwrap to a scalar
    # string, and `@recorded` would then splat it character by character.
    $recorded = @(if ($names) { $names } else { (Get-ChildItem "$OutputDir\*.webp").BaseName })
    foreach ($n in $recorded) { Copy-Item "$OutputDir\$n.webp" $orig -Force }
    Write-Host "==> Retiming ${PlaybackFactor}x slower..." -ForegroundColor Cyan
    python "$DemosDir\retime.py" $PlaybackFactor @recorded
    if ($LASTEXITCODE -ne 0) { throw "retime failed" }
}
