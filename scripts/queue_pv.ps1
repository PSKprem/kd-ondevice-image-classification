# P25 PlantVillage queue (5 runs). Waits until the CIFAR-100 queue (queue_v2_exp4.py) has finished,
# so it can be started at any time and will not compete for the GPU. Resumable: run it again after any interruption.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\queue_pv.ps1 [-Epochs 20]
param([int]$Epochs = 20)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "queue_v2_exp4\.py" }) {
    Write-Host "$(Get-Date -Format HH:mm:ss) waiting for queue_v2_exp4.py to finish ..."
    Start-Sleep -Seconds 60
}

& $py -u scripts/queue_pv.py --epochs $Epochs 2>&1 | Tee-Object -FilePath "$logs\queue_pv.log" -Append

Write-Host "=== QUEUE DONE ==="
