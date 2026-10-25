# P25 low-data study (8 runs, ~30 min): w0.5 / w1.0 on 10% and 25% of the CIFAR-100 train split, baseline vs KD.
# Resumable: run it again after any interruption.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_lowdata.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/queue_lowdata.py 2>&1 | Tee-Object -FilePath "$logs\queue_lowdata.log" -Append

Write-Host "=== QUEUE DONE ==="
