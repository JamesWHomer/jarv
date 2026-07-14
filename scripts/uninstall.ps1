param(
    [string]$InstallDir = "$env:LOCALAPPDATA\Programs\Jarv",
    [switch]$Purge
)

$ErrorActionPreference = "Stop"
$InstallDir = [IO.Path]::GetFullPath($InstallDir).TrimEnd('\', '/')
$Executable = Join-Path $InstallDir "jarv.exe"

$RunningJarv = Get-Process -Name "jarv" -ErrorAction SilentlyContinue | Where-Object {
    try {
        $ProcessPath = $_.Path
        $ProcessPath -and (
            [string]::Equals($ProcessPath, $Executable, [StringComparison]::OrdinalIgnoreCase) -or
            $ProcessPath.StartsWith("$InstallDir\", [StringComparison]::OrdinalIgnoreCase)
        )
    } catch {
        $false
    }
}
if ($RunningJarv) {
    throw "Jarv is still running from $InstallDir. Exit it and run this script again."
}

Remove-Item -LiteralPath $Executable -Force -ErrorAction SilentlyContinue
if ((Test-Path -LiteralPath $InstallDir) -and
    -not (Get-ChildItem -LiteralPath $InstallDir -Force | Select-Object -First 1)) {
    Remove-Item -LiteralPath $InstallDir -Force
}

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$Wanted = $InstallDir.TrimEnd('\', '/').Trim()
$PathParts = @($UserPath -split ";" | Where-Object {
    $_ -and -not [string]::Equals($_.TrimEnd('\', '/').Trim(), $Wanted, [StringComparison]::OrdinalIgnoreCase)
})
# This matches install.ps1; SetEnvironmentVariable may expand REG_EXPAND_SZ entries.
$NewPath = $PathParts -join ";"
if ($NewPath -ne $UserPath) {
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
}

if ($Purge) {
    $Answer = Read-Host "Delete user data in $HOME\.jarv and cached clipboard images? [y/N]"
    if ($Answer -match '^(y|yes)$') {
        Remove-Item -LiteralPath (Join-Path $HOME ".jarv") -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath (Join-Path ([IO.Path]::GetTempPath()) "jarv-clipboard") -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Removed Jarv user data."
    } else {
        Write-Host "Kept Jarv user data."
    }
}

Write-Host "Uninstalled Jarv from $InstallDir."
