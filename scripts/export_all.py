"""P25 — export every model to TFLite and measure the exported accuracy on the CIFAR-100 test set.

Run with the EXPORT venv python (D:\\kd-project\\.venv-export). For each checkpoint:
  1. export.py --onnx-only          -> model.onnx
  2. onnx2tf -b 1                   -> tf/model_float32.tflite
  3. quantize_tflite.py (dynamic)   -> model_int8_dynamic.tflite
  4. eval_tflite.py                 -> one row per file in results/export_eval_v2.csv

Models: KD student (the model that would ship) + baseline student for each of w0.5 / w1.0 / w1.4 / MBV3-Small
(baseline and KD have identical size/latency, only accuracy differs) + the ResNet-50 teacher (FP32 only).
Accuracy: INT8 on the full 10k test set (agreement with PyTorch included); FP32 on 2000 images (it is identical to
PyTorch by construction; agreement checked). Resumable: existing files / CSV rows are skipped.

Usage (from repo root, export venv):
    D:\\kd-project\\.venv-export\\Scripts\\python.exe -u scripts/export_all.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("D:/kd-project/runs_v2")
EXPORT = Path("D:/kd-project/export")
EVAL_CSV = ROOT / "results" / "export_eval_v2.csv"
VENV_SCRIPTS = Path(sys.executable).parent

# (run name, role, config, extra export/eval args)
ITEMS = [
    ("kd_cifar100_mbv3s_v2_adamw1e-3_T2_a0.9", "kd", "configs/student.yaml", ["--model", "mobilenetv3_small"]),
    ("student_mbv3s_cifar100_v2_adamw1e-3", "baseline", "configs/student.yaml", ["--model", "mobilenetv3_small"]),
    ("kd_cifar100_w0.5_v2_adamw1e-3_T2_a0.9", "kd", "configs/student.yaml", ["--width-mult", "0.5"]),
    ("student_mbv2_cifar100_w0.5_v2_adamw1e-3", "baseline", "configs/student.yaml", ["--width-mult", "0.5"]),
    ("kd_cifar100_w1.0_v2_sgd0.01_T2_a0.9", "kd", "configs/student.yaml", ["--width-mult", "1.0"]),
    ("student_mbv2_cifar100_w1.0_v2_sgd0.01", "baseline", "configs/student.yaml", ["--width-mult", "1.0"]),
    ("kd_cifar100_w1.4_v2_sgd0.01_T2_a0.9", "kd", "configs/student.yaml", ["--width-mult", "1.4"]),
    ("student_mbv2_cifar100_w1.4_v2_sgd0.01", "baseline", "configs/student.yaml", ["--width-mult", "1.4"]),
    ("teacher_resnet50_cifar100_v2", "teacher", "configs/teacher.yaml", ["--model", "resnet50"]),
]

DRY = False


def log(msg: str) -> None:
    print(f"[export {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env() -> dict:
    e = dict(os.environ)
    e["PATH"] = f"{VENV_SCRIPTS};{e.get('PATH', '')}"
    e["TEMP"] = e["TMP"] = "D:\\kd-project\\tmp"
    e["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return e


def run(cmd: list[str]) -> None:
    log("RUN " + " ".join(str(c) for c in cmd)[:230])
    if DRY:
        return
    rc = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env()).returncode
    if rc != 0:
        raise RuntimeError(f"command failed (rc={rc}): {cmd[:3]}")


def done_rows() -> set[tuple[str, str]]:
    if not EVAL_CSV.exists():
        return set()
    with open(EVAL_CSV, encoding="utf-8") as f:
        return {(r["tag"], r["tflite"]) for r in csv.DictReader(f)}


def convert(name: str, config: str, extra: list[str]) -> Path:
    out = EXPORT / name
    ckpt = RUNS / name / "best.pt"
    if not ckpt.exists() and not DRY:
        raise FileNotFoundError(ckpt)
    if not (out / "model.onnx").exists():
        run([sys.executable, "scripts/export.py", "--ckpt", ckpt, "--config", config, *extra,
             "--out", out, "--onnx-only"])
    fp32 = out / "tf" / "model_float32.tflite"
    if not fp32.exists():
        run([VENV_SCRIPTS / "onnx2tf.exe", "-i", out / "model.onnx", "-o", out / "tf", "-b", "1"])
    int8 = out / "model_int8_dynamic.tflite"
    if not int8.exists():
        run([sys.executable, "scripts/quantize_tflite.py", "--fp32", fp32, "--out", int8, "--mode", "dynamic"])
    return out


def evaluate(name: str, tflite: Path, config: str, extra: list[str], n: int, with_ckpt: bool) -> None:
    if (name, tflite.name) in done_rows():
        log(f"SKIP eval {name}/{tflite.name} (already in {EVAL_CSV.name})")
        return
    cmd = [sys.executable, "scripts/eval_tflite.py", "--tflite", tflite, "--config", config, *extra,
           "--n", str(n), "--tag", name, "--csv", EVAL_CSV]
    if with_ckpt:
        cmd += ["--ckpt", RUNS / name / "best.pt"]
    run(cmd)


def main() -> None:
    global DRY
    ap = argparse.ArgumentParser(description="P25 export + exported-accuracy check for all models")
    ap.add_argument("--dry-run", action="store_true")
    DRY = ap.parse_args().dry_run
    t0 = time.time()
    for name, role, config, extra in ITEMS:
        log(f"===== {name} ({role}) =====")
        try:
            out = convert(name, config, extra)
            if role == "teacher":
                evaluate(name, out / "tf" / "model_float32.tflite", config, extra, 300, True)
                continue
            if role == "kd":
                evaluate(name, out / "tf" / "model_float32.tflite", config, extra, 2000, True)
            evaluate(name, out / "model_int8_dynamic.tflite", config, extra, 0, True)
        except Exception as e:  # keep going with the other models
            log(f"FAILED {name}: {e}")
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min. Results: {EVAL_CSV}")


if __name__ == "__main__":
    main()
