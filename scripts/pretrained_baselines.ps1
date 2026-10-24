# P25: pretrained (transfer-learning) baselines, no KD, for the students that
# have ImageNet weights: MobileNetV2 w0.5 / w1.4 (timm) and MobileNetV3-Small.
# (MobileNetV2 w1.0 was already trained on Night 1.)
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts\pretrained_baselines.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$logs = "D:\kd-project\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

foreach ($w in "0.5", "1.4") {
    Write-Host "=== student baseline MobileNetV2 width $w (pretrained) ==="
    & $py -u -m src.train --config configs/student.yaml --width-mult $w --run-name "student_mbv2_cifar100_w$w" 2>&1 | Tee-Object -FilePath "$logs\student_w$w.log"
}

Write-Host "=== student baseline MobileNetV3-Small (pretrained) ==="
& $py -u -m src.train --config configs/student.yaml --model mobilenetv3_small --run-name "student_mbv3s_cifar100" 2>&1 | Tee-Object -FilePath "$logs\student_mbv3s.log"

Write-Host "=== ALL DONE ==="
