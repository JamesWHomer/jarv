# Shared setup for record.ps1 (single/sequential) and record-all.ps1 (parallel).
# Dot-source it (`. "$PSScriptRoot\_record-common.ps1"`): it defines the repo
# paths, the ffmpeg-shim / PATH bootstrap, the per-tape reasoning-effort map, and
# the retime step in the caller's scope. See demos/README.md for requirements.
$ErrorActionPreference = 'Stop'

$DemosDir  = $PSScriptRoot
$RepoRoot  = Split-Path $DemosDir -Parent
$OutputDir = "$DemosDir\output"

# VHS bakes the tapes' timing (TypingSpeed 40ms etc.) into the WebP, which plays
# too fast. After recording we rescale every frame delay by this factor
# (>1 = slower) via retime.py — same frames, same file size, only slower.
$PlaybackFactor = 1.4

# High reasoning effort parks demos on a spinner for minutes. Trivial-question
# tapes record with reasoning disabled entirely; the rest use low. Effort is a
# global config value (config.json), so record-all groups tapes into one parallel
# wave per effort; record.ps1 just sets it per tape. Either way the user's
# setting is restored afterwards.
$TapeEffort    = @{ oneshot = 'none'; undo = 'none' }
$DefaultEffort = 'low'
function Get-TapeEffort([string]$name) {
    if ($TapeEffort[$name]) { $TapeEffort[$name] } else { $DefaultEffort }
}

# Put vhs/ttyd/ffmpeg-shim on PATH and compile the shim. Idempotent — safe to
# call once per script run.
function Initialize-RecordEnv {
    # winget portable installs may not be on PATH in this shell yet.
    $wingetPkgs = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    foreach ($dir in @(
        "$wingetPkgs\charmbracelet.vhs_Microsoft.Winget.Source_8wekyb3d8bbwe\vhs_0.11.0_Windows_x86_64",
        "$wingetPkgs\tsl0922.ttyd_Microsoft.Winget.Source_8wekyb3d8bbwe"
    )) {
        if (Test-Path $dir) { $env:Path = "$dir;$env:Path" }
    }
    foreach ($bin in 'vhs', 'ttyd', 'ffmpeg', 'jarv') {
        if (-not (Get-Command $bin -ErrorAction SilentlyContinue)) {
            throw "'$bin' not found on PATH - see the requirements in demos/README.md."
        }
    }

    # Resolve the real ffmpeg BEFORE the shim shadows it, then compile the shim if
    # the source is newer and prepend it so vhs picks it up. VHS hands ffmpeg no
    # codec options, so .webp output defaults to lossy VP8 (4:2:0 chroma smears
    # colored text); the shim upgrades .webp encodes to lossless RGB (VP8L).
    $realFfmpeg = (Get-Command ffmpeg).Source
    $shimSrc = "$DemosDir\bin\ffmpeg-shim.cs"
    $shimExe = "$DemosDir\bin\shim\ffmpeg.exe"
    if (-not (Test-Path $shimExe) -or (Get-Item $shimSrc).LastWriteTime -gt (Get-Item $shimExe).LastWriteTime) {
        New-Item -ItemType Directory -Force "$DemosDir\bin\shim" | Out-Null
        & "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe" -nologo -out:$shimExe $shimSrc
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg shim compile failed" }
    }
    $env:FFMPEG_SHIM_REAL = $realFfmpeg
    $env:FFMPEG_SHIM_LOG = "$DemosDir\bin\ffmpeg-shim.log"
    $env:Path = "$DemosDir\bin\shim;$env:Path"

    # Record from the repo root so demos show the real project (cwd in the footer,
    # `cat README.md`, git context). Tapes write to demos/output/ accordingly.
    Set-Location $RepoRoot
    New-Item -ItemType Directory -Force $OutputDir | Out-Null
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
    uv run python "$DemosDir\retime.py" $PlaybackFactor @recorded
    if ($LASTEXITCODE -ne 0) { throw "retime failed" }
}
