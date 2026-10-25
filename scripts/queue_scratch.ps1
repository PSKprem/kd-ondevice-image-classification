# P25 from-scratch comparison (4 runs, ~1 h): MBV2 w0.5 / w1.0 without ImageNet weights, baseline vs KD.
# Resumable: run it again after any interruption.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_scratch.ps1 [-Epochs 20]
param([int]$Epochs = 20)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/queue_scratch.py --epochs $Epochs 2>&1 | Tee-Object -FilePath "$logs\queue_scratch.log" -Append

Write-Host "=== QUEUE DONE ==="
