param(
    [string]$Version = "latest",
    [string]$InstallDir = "$env:LOCALAPPDATA\Programs\Jarv"
)

$ErrorActionPreference = "Stop"
$Owner = "JamesWHomer"
$Repo = "jarv"

if ([Environment]::Is64BitOperatingSystem -eq $false) {
    throw "Jarv standalone builds require a 64-bit Windows system."
}

$Platform = "windows"
$Machine = $env:PROCESSOR_ARCHITECTURE.ToLowerInvariant()
switch ($Machine) {
    "amd64" { $Arch = "x86_64" }
    "arm64" { $Arch = "arm64" }
    default { throw "Unsupported architecture: $Machine" }
}

if ($Version -eq "latest") {
    $ManifestUrl = "https://github.com/$Owner/$Repo/releases/latest/download/release-manifest.json"
} else {
    $Tag = if ($Version.StartsWith("v")) { $Version } else { "v$Version" }
    $ManifestUrl = "https://github.com/$Owner/$Repo/releases/download/$Tag/release-manifest.json"
}

$TempDir = Join-Path ([IO.Path]::GetTempPath()) ("jarv-install-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $TempDir | Out-Null

try {
    $ManifestPath = Join-Path $TempDir "release-manifest.json"
    Invoke-WebRequest -Uri $ManifestUrl -OutFile $ManifestPath
    $Manifest = Get-Content -Raw $ManifestPath | ConvertFrom-Json
    $Asset = $Manifest.assets | Where-Object {
        $_.platform -eq $Platform -and $_.architecture -eq $Arch
    } | Select-Object -First 1
    if (-not $Asset) {
        throw "No jarv release asset for $Platform/$Arch"
    }

    $ArchivePath = Join-Path $TempDir $Asset.name
    Invoke-WebRequest -Uri $Asset.download_url -OutFile $ArchivePath
    $ActualSha = (Get-FileHash -Algorithm SHA256 -Path $ArchivePath).Hash.ToLowerInvariant()
    if ($ActualSha -ne $Asset.sha256.ToLowerInvariant()) {
        throw "Checksum mismatch for $($Asset.name): expected $($Asset.sha256), got $ActualSha"
    }

    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $TempDir -Force
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $TempDir "jarv.exe") -Destination (Join-Path $InstallDir "jarv.exe") -Force

    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $PathParts = @($UserPath -split ";" | Where-Object { $_ })
    if ($PathParts -notcontains $InstallDir) {
        $NewPath = (@($PathParts) + $InstallDir) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
        $env:Path = "$env:Path;$InstallDir"
        Write-Host "Added $InstallDir to the user PATH."
    }

    & (Join-Path $InstallDir "jarv.exe") --version
} finally {
    Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}

