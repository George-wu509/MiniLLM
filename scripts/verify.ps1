param(
    [ValidateSet('Fast', 'CPU', 'GPU')][string]$Profile = 'Fast',
    [string]$Python = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $labRoot '.venv\Scripts\python.exe' }
$labArguments = @((Join-Path $PSScriptRoot 'verify.py'), '--profile', $Profile.ToLowerInvariant())
if ($Output) { $labArguments += @('--output', $Output) }
& $Python @labArguments
if ($LASTEXITCODE -ne 0) { throw "Verification failed with exit code $LASTEXITCODE. Inspect the evidence logs." }
