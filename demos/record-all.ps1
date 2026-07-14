# Re-record demo animations from the .tape scripts in tapes/, in PARALLEL.
# Usage:
#   .\record-all.ps1              # record every tape
#   .\record-all.ps1 hero usage   # record only the named tapes
# To record a single tape sequentially, use record.ps1 or the per-tape wrappers
# in tapes\ (e.g. tapes\hero.ps1). Shared setup lives in _record-common.ps1;
# see demos/README.md for requirements.
#
# Tapes record grouped into waves by reasoning effort (the effort lives in the
# shared config.json, so it can only vary between waves, not within one). Total
# wall time is roughly the longest tape per wave.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Names)

. "$PSScriptRoot\_record-common.ps1"
Initialize-RecordEnv

$tapes = Get-ChildItem "$DemosDir\tapes\*.tape"
if ($Names) {
    $tapes = $tapes | Where-Object { $Names -contains $_.BaseName }
    if (-not $tapes) { throw "No tapes matched: $Names" }
}

$prevEffort = Get-CurrentEffort

$recordJob = {
    param($tapePath, $repoRoot, $pathEnv, $shimReal, $shimLog)
    $env:Path = $pathEnv
    $env:FFMPEG_SHIM_REAL = $shimReal
    $env:FFMPEG_SHIM_LOG = $shimLog
    # Fresh terminal identity per take: jarv keys sessions on WT_SESSION, so this
    # keeps each recording's history isolated from ours and from other takes.
    $env:WT_SESSION = [guid]::NewGuid().ToString()
    Set-Location $repoRoot
    vhs $tapePath | Out-Null
    $LASTEXITCODE
}

$failed = @()
try {
    # commands.tape cycles reasoning_effort on camera mid-wave; concurrent takes
    # read config at launch (before the cycling lands), and retries re-assert the
    # wave's effort first.
    $waves = $tapes | Group-Object { Get-TapeEffort $_.BaseName }
    foreach ($wave in $waves) {
        Set-Effort $wave.Name
        Write-Host "==> Recording in parallel (reasoning_effort $($wave.Name)): $(($wave.Group | ForEach-Object BaseName) -join ', ')" -ForegroundColor Cyan
        $jobs = @{}
        foreach ($tape in $wave.Group) {
            $jobs[$tape.BaseName] = Start-Job -ScriptBlock $recordJob -ArgumentList `
                $tape.FullName, $RepoRoot, $env:Path, $env:FFMPEG_SHIM_REAL, $env:FFMPEG_SHIM_LOG
        }
        # Tapes bound their own waits (240s max) — 10 minutes means a hang.
        Wait-Job -Job @($jobs.Values) -Timeout 600 | Out-Null
        foreach ($name in $jobs.Keys) {
            $job = $jobs[$name]
            if ($job.State -ne 'Completed') { Stop-Job $job }
            $exit = @(Receive-Job $job -ErrorAction SilentlyContinue)[-1]
            Remove-Job $job -Force
            if ($exit -ne 0) {
                Write-Host "    $name failed (exit $exit)" -ForegroundColor Yellow
                $failed += $name
            }
            else {
                Write-Host "    $name done" -ForegroundColor Green
            }
        }
    }

    # One sequential retry per failed tape: the first heads-up launch after an
    # idle stretch sometimes comes up with dead keyboard input (the tapes' Wait
    # patterns turn that into a loud timeout), and ttyd itself occasionally
    # flakes. Effort is re-asserted per retry because commands.tape may have
    # cycled it during its wave.
    $stillFailed = @()
    foreach ($name in $failed) {
        Set-Effort (Get-TapeEffort $name)
        Write-Host "==> Retrying $name (reasoning_effort $(Get-TapeEffort $name))..." -ForegroundColor Cyan
        $env:WT_SESSION = [guid]::NewGuid().ToString()
        vhs "$DemosDir\tapes\$name.tape"
        if ($LASTEXITCODE -ne 0) { $stillFailed += $name }
    }
    $failed = $stillFailed
}
finally {
    Restore-Effort $prevEffort
}

if ($failed) {
    Get-ChildItem "$OutputDir\*.webp" | Format-Table Name, @{L = 'Size'; E = { '{0:N0} KB' -f ($_.Length / 1KB) } }, LastWriteTime
    throw "Failed tapes: $($failed -join ', ')"
}

$recorded = if ($Names) { $Names } else { (Get-ChildItem "$OutputDir\*.webp").BaseName }
Complete-Retime $recorded

Write-Host ""
Get-ChildItem "$OutputDir\*.webp" | Format-Table Name, @{L = 'Size'; E = { '{0:N0} KB' -f ($_.Length / 1KB) } }, LastWriteTime
Write-Host "Done. Eyeball each animation in output\ before running publish.ps1." -ForegroundColor Green
