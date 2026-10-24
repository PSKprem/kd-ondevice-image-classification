# P25 Night 2: KD T x alpha sweep on MobileNetV2 w0.5 (pretrained), 12 runs.
# Resumable: re-running skips (T, alpha) pairs already in results/results.csv.
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\sweep_night2.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

& $py -u scripts/sweep.py --config configs/kd.yaml `
    --teacher-ckpt D:/kd-project/runs/teacher_resnet50_cifar100/best.pt `
    --width-mult 0.5 2>&1 | Tee-Object -FilePath "$logs\sweep_w0.5.log"

Write-Host "=== SWEEP DONE ==="
