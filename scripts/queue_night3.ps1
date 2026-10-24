# P25 Night 3 queue: teacher (no label smoothing) -> KD test w1.0 -> w0.5 recipe test -> KD re-sweep.
# Runs all steps one after another, then stops. Resumable: just run it again.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_night3.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/queue_night3.py 2>&1 | Tee-Object -FilePath "$logs\queue_night3.log" -Append

Write-Host "=== QUEUE DONE ==="
