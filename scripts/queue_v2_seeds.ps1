# P25 v2 seed check: baseline + KD (T=2, alpha=0.9) with seeds 0 and 1 on w0.5 and w1.0 (8 runs, ~1.7 h).
# Resumable: run it again after any interruption. Updates results/kd_gain_v2.csv at the end.
# Other students: add  --seed-students w1.4 mbv3s  (see scripts/queue_v2_exp4.py)
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_v2_seeds.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/queue_v2_exp4.py --seeds 2>&1 | Tee-Object -FilePath "$logs\queue_v2_seeds.log" -Append

Write-Host "=== QUEUE DONE ==="
