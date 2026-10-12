param([string]$Run = 'smoke', [int]$Port = 6006)
$ErrorActionPreference = 'Stop'
$labRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
& "$labRoot\.venv\Scripts\python.exe" -m tensorboard.main --logdir "$labRoot\artifacts\$Run\tensorboard" --host 127.0.0.1 --port $Port
