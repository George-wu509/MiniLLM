$ErrorActionPreference = 'Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
$labPython = Join-Path $labRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) { throw 'Run scripts\setup.ps1 first.' }
$env:PYTHONUTF8 = '1'
Push-Location $labRoot
try {
    & $labPython -m minillm @args
    if ($LASTEXITCODE -ne 0) { throw "MiniLLM exited with code $LASTEXITCODE" }
} finally { Pop-Location }
