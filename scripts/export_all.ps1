# P25 export of all models to TFLite (FP32 + INT8 dynamic) + exported-accuracy check (CPU, export venv, ~1-1.5 h).
# Resumable: run it again after any interruption.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\export_all.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "D:\kd-project\.venv-export\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/export_all.py 2>&1 | Tee-Object -FilePath "$logs\export_all.log" -Append

Write-Host "=== EXPORT DONE ==="
