# P25 v2 seed check: baseline + KD (T=2, alpha=0.9) with seeds 0 and 1 for the chosen students (default w0.5 w1.0, ~1.7 h;
# w1.4 + mbv3s ~1.3 h). Resumable: run it again after any interruption. Updates results/kd_gain_v2.csv at the end.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_v2_seeds.ps1 [-Students "w1.4 mbv3s"]
param([string]$Students = "w0.5 w1.0")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

$list = $Students -split "\s+" | Where-Object { $_ }
& $py -u scripts/queue_v2_exp4.py --seeds --seed-students @list 2>&1 | Tee-Object -FilePath "$logs\queue_v2_seeds.log" -Append

Write-Host "=== QUEUE DONE ==="
