param([string]$Run = ('smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\lab.ps1" all --run $Run --config configs/smoke.json
& "$PSScriptRoot\lab.ps1" nanogpt --run $Run
& "$PSScriptRoot\lab.ps1" evaluate --run $Run
Write-Host "Artifacts: artifacts\$Run"
