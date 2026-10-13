param(
    [string]$Python = '',
    [switch]$CpuOnly,
    [switch]$Development
)
$ErrorActionPreference = 'Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
Push-Location $labRoot
try {
    if (-not $Python) {
        $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundledPython) { $Python = $bundledPython }
        else { $Python = 'python' }
    }
    & $Python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
    if ($LASTEXITCODE -ne 0) { throw 'Supply a real Python 3.12 executable with -Python.' }
    & $Python scripts/bootstrap_upstreams.py
    if ($LASTEXITCODE -ne 0) { throw 'Pinned upstream preparation failed.' }
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment setup failed.' }
    $index = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
    & .\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url $index
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    $requirements = if ($CpuOnly) { 'requirements-cpu.lock.txt' } else { 'requirements.lock.txt' }
    & .\.venv\Scripts\python.exe -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    if ($Development) {
        & .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
        if ($LASTEXITCODE -ne 0) { throw 'Development dependency installation failed.' }
    }
    & .\.venv\Scripts\python.exe -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
    & .\.venv\Scripts\python.exe -m minillm doctor
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
} finally { Pop-Location }
