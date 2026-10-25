# P25 v2 Experiment 4 queue: KD (best T/alpha) on all students + seed check (w0.5, w1.0).
# Resumable: run it again after any interruption.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_v2_exp4.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/queue_v2_exp4.py 2>&1 | Tee-Object -FilePath "$logs\queue_v2_exp4.log" -Append

Write-Host "=== QUEUE DONE ==="
