# P25 v2 Night 1 queue (Tasks 3-5): teacher -> 8 baselines -> full T x alpha sweep on w1.0.
# Resumable: run it again after any interruption.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_v2_night1.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/queue_v2_night1.py 2>&1 | Tee-Object -FilePath "$logs\queue_v2_night1.log" -Append

Write-Host "=== QUEUE DONE ==="
