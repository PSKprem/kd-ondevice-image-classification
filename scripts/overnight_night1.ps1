# P25 Night 1 overnight run: ResNet-50 teacher, then 4 MobileNetV2 baselines (no KD).
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\overnight_night1.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

Write-Host "=== teacher (ResNet-50) ==="
& $py -u -m src.train --config configs/teacher.yaml 2>&1 | Tee-Object -FilePath "$logs\teacher.log"

foreach ($w in "1.0", "0.75", "0.5", "0.35") {
    Write-Host "=== student baseline width $w ==="
    & $py -u -m src.train --config configs/student.yaml --width-mult $w --run-name "student_mbv2_cifar100_w$w" 2>&1 | Tee-Object -FilePath "$logs\student_w$w.log"
}

Write-Host "=== ALL DONE ==="
