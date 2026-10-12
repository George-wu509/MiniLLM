param(
    [string]$Python = '',
    [switch]$CpuOnly
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
    $lock = Get-Content -LiteralPath 'upstream-lock.json' -Raw | ConvertFrom-Json
    foreach ($name in @('minimind', 'nanogpt')) {
        $entry = $lock.$name
        $repoPath = Join-Path $labRoot "vendor\$name"
        if (-not (Test-Path -LiteralPath $repoPath)) {
            git clone --no-checkout --depth 1 $entry.url $repoPath
            if ($LASTEXITCODE -ne 0) { throw "Clone failed: $name" }
            git -C $repoPath fetch --depth 1 origin $entry.commit
            if ($LASTEXITCODE -ne 0) { throw "Pinned commit fetch failed: $name" }
            git -C $repoPath checkout --detach $entry.commit
            if ($LASTEXITCODE -ne 0) { throw "Checkout failed: $name" }
        }
        $head = git -C $repoPath rev-parse HEAD
        if ($head -ne $entry.commit) { throw "$name HEAD differs from upstream-lock.json; preserve changes and use the pinned source." }
    }
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment setup failed.' }
    $index = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
    & .\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url $index
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    $requirements = if ((Test-Path -LiteralPath 'requirements.lock.txt') -and -not $CpuOnly) { 'requirements.lock.txt' } else { 'requirements.txt' }
    & .\.venv\Scripts\python.exe -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & .\.venv\Scripts\python.exe -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
    & .\.venv\Scripts\python.exe -m minillm doctor
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
} finally { Pop-Location }
