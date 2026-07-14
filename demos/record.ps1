# Record one or more demo tapes individually, sequentially — the full pipeline
# per tape (lossless capture, per-tape reasoning effort, retime). For the fast
# parallel path over everything at once, use record-all.ps1 instead.
#
# Usage:
#   .\record.ps1 hero            # record just hero
#   .\record.ps1 hero commands   # record these, in order
# The per-tape wrappers in tapes\ (e.g. tapes\hero.ps1) call this.
param([Parameter(Mandatory, ValueFromRemainingArguments = $true)][string[]]$Names)

. "$PSScriptRoot\_record-common.ps1"
Initialize-RecordEnv

$prevEffort = Get-CurrentEffort
$recorded = @()
$failed = @()
try {
    foreach ($name in $Names) {
        $tape = "$DemosDir\tapes\$name.tape"
        if (-not (Test-Path $tape)) { throw "No tape named '$name' in tapes\" }
        $effort = Get-TapeEffort $name
        Set-Effort $effort
        Write-Host "==> Recording $name (reasoning_effort $effort)..." -ForegroundColor Cyan
        # Fresh terminal identity per take: jarv keys session history on
        # WT_SESSION, so this isolates the recording from ours.
        $env:WT_SESSION = [guid]::NewGuid().ToString()
        vhs $tape
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    $name failed" -ForegroundColor Yellow
            $failed += $name
        }
        else {
            Write-Host "    $name done" -ForegroundColor Green
            $recorded += $name
        }
    }
}
finally {
    Restore-Effort $prevEffort
}

if ($failed) { throw "Failed tapes: $($failed -join ', ')" }

Complete-Retime $recorded

Write-Host ""
Get-ChildItem "$OutputDir\*.webp" | Where-Object { $recorded -contains $_.BaseName } |
    Format-Table Name, @{L = 'Size'; E = { '{0:N0} KB' -f ($_.Length / 1KB) } }, LastWriteTime
Write-Host "Done. Eyeball output\ before running publish.ps1." -ForegroundColor Green
