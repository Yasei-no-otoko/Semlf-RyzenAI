[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8008
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$sdkRoot = 'C:\Program Files\RyzenAI\1.8.0'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Project .venv is missing. Follow docs/RYZENAI.md first.'
}
if (-not (Test-Path -LiteralPath $sdkRoot -PathType Container)) {
    throw "Ryzen AI 1.8.0 SDK is missing: $sdkRoot"
}
$previousSdk = $env:RYZEN_AI_INSTALLATION_PATH
$previousUtf8 = $env:PYTHONUTF8
$exitCode = 1
Push-Location $PSScriptRoot
try {
    $env:RYZEN_AI_INSTALLATION_PATH = $sdkRoot
    $env:PYTHONUTF8 = '1'
    & $python -m semif_phase1.ryzenai_demo --port $Port
    $exitCode = $LASTEXITCODE
} finally {
    $env:RYZEN_AI_INSTALLATION_PATH = $previousSdk
    $env:PYTHONUTF8 = $previousUtf8
    Pop-Location
}
exit $exitCode
