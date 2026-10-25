# P25 optimizer de-confound: KD (T=2, alpha=0.9) with the other recipe for every student (4 runs, ~1.1 h).
# Resumable: run it again after any interruption.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_recipe_swap.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/queue_recipe_swap.py 2>&1 | Tee-Object -FilePath "$logs\queue_recipe_swap.log" -Append

Write-Host "=== QUEUE DONE ==="
