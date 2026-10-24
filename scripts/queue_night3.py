"""P25 Night 3 queue: fix the label-smoothing teacher + w0.5 recipe, then re-sweep KD.

Runs everything back to back, unattended, and stops when done:
  1. Teacher ResNet-50 retrained WITHOUT label smoothing (label_smoothing=0).
  2. KD test on MBV2 w1.0 (T=4, alpha=0.9) with the new teacher,
     + fairness baseline w1.0 without label smoothing (same recipe, no KD).
  3. MBV2 w0.5 quick recipe test (4 epochs each, logged to results/quick_tests.csv),
     then a full 20-epoch w0.5 baseline with the winning recipe.
  4. Small KD re-sweep T in {2,4} x alpha in {0.5,0.9} on the best working small
     student: w0.5 (winning recipe) if its new baseline >= W05_OK_THRESHOLD,
     otherwise MobileNetV3-Small (+ its own no-label-smoothing baseline).

Resumable: any run whose run_name is already in the results CSV is skipped, and an
interrupted run continues from its last.pt (--resume). Old runs are never touched.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_night3.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import read_results  # noqa: E402

RESULTS_CSV = "results/results.csv"
QUICK_CSV = "results/quick_tests.csv"
RUNS_DIR = Path("D:/kd-project/runs")

TEACHER_RUN = "teacher_resnet50_cifar100_nols"
TEACHER_CKPT = RUNS_DIR / TEACHER_RUN / "best.pt"

QUICK_EPOCHS = 4
W05_OK_THRESHOLD = 68.0     # w0.5 counts as "fixed" if the full baseline reaches this
SWEEP_T = [2.0, 4.0]
SWEEP_ALPHA = [0.5, 0.9]

# Candidate recipes for the w0.5 student (all with label_smoothing=0, warmup 1).
W05_RECIPES = {
    "sgd0.01": ["--optimizer", "sgd", "--lr", "0.01", "--weight-decay", "0.0005"],
    "sgd0.05": ["--optimizer", "sgd", "--lr", "0.05", "--weight-decay", "0.0005"],
    "sgd0.1": ["--optimizer", "sgd", "--lr", "0.1", "--weight-decay", "0.0005"],
    "adamw1e-3": ["--optimizer", "adamw", "--lr", "0.001", "--weight-decay", "0.05"],
}
DEFAULT_RECIPE = W05_RECIPES["sgd0.01"]   # same as all earlier runs

STATUS: list[str] = []


def log(msg: str) -> None:
    print(f"[queue {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def row_for(run_name: str, csv_path: str = RESULTS_CSV) -> dict | None:
    if not (ROOT / csv_path).exists():
        return None
    for r in read_results(str(ROOT / csv_path)):
        if r.get("run_name") == run_name:
            return r
    return None


def top1(run_name: str, csv_path: str = RESULTS_CSV, key: str = "best_top1") -> float | None:
    r = row_for(run_name, csv_path)
    try:
        return float(r[key]) if r else None
    except (TypeError, ValueError, KeyError):
        return None


def train(run_name: str, config: str, extra: list[str],
          csv_path: str = RESULTS_CSV) -> bool:
    """Run one training job unless it's already logged. Returns True on success."""
    if row_for(run_name, csv_path) is not None:
        log(f"SKIP {run_name} (already in {csv_path})")
        STATUS.append(f"skip  {run_name}")
        return True
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", config,
           "--run-name", run_name, "--label-smoothing", "0",
           "--results-csv", csv_path, "--resume", *extra]
    log(f"RUN  {run_name}\n      {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    mins = (time.time() - t0) / 60
    ok = rc == 0 and row_for(run_name, csv_path) is not None
    STATUS.append(f"{'ok  ' if ok else 'FAIL'}  {run_name}  ({mins:.1f} min)")
    log(f"{'DONE' if ok else 'FAILED (rc=' + str(rc) + ')'} {run_name} in {mins:.1f} min")
    return ok


def kd_args(T: float, alpha: float) -> list[str]:
    return ["--teacher-ckpt", str(TEACHER_CKPT).replace("\\", "/"),
            "--temperature", str(T), "--alpha", str(alpha)]


def main() -> None:
    t_start = time.time()

    # ---- Step 1: teacher without label smoothing ----
    log("STEP 1/4: teacher ResNet-50, label_smoothing=0")
    train(TEACHER_RUN, "configs/teacher.yaml", DEFAULT_RECIPE)
    teacher_ok = TEACHER_CKPT.exists()
    if not teacher_ok:
        log(f"Teacher checkpoint missing ({TEACHER_CKPT}); KD steps will be skipped.")

    # ---- Step 2: KD test on w1.0 + fair no-LS baseline ----
    log("STEP 2/4: KD test on MBV2 w1.0 (T=4, alpha=0.9) + no-LS baseline")
    if teacher_ok:
        train("kd_cifar100_w1.0_T4.0_a0.9_nols", "configs/kd.yaml",
              ["--width-mult", "1.0", *DEFAULT_RECIPE, *kd_args(4.0, 0.9)])
    train("student_mbv2_cifar100_w1.0_nols", "configs/student.yaml",
          ["--width-mult", "1.0", *DEFAULT_RECIPE])

    # ---- Step 3: w0.5 recipe quick test, then full baseline with the winner ----
    log(f"STEP 3/4: w0.5 recipe quick test ({QUICK_EPOCHS} epochs each)")
    for name, recipe in W05_RECIPES.items():
        train(f"qt_w0.5_{name}", "configs/student.yaml",
              ["--width-mult", "0.5", "--epochs", str(QUICK_EPOCHS), *recipe],
              csv_path=QUICK_CSV)
    scores = {n: top1(f"qt_w0.5_{n}", QUICK_CSV, "final_top1") for n in W05_RECIPES}
    log("quick-test final val@1: " + ", ".join(f"{n}={s}" for n, s in scores.items()))
    valid = {n: s for n, s in scores.items() if s is not None}
    best_recipe = max(valid, key=valid.get) if valid else "sgd0.01"
    log(f"winning w0.5 recipe: {best_recipe}")
    w05_base_run = f"student_mbv2_cifar100_w0.5_nols_{best_recipe}"
    train(w05_base_run, "configs/student.yaml",
          ["--width-mult", "0.5", *W05_RECIPES[best_recipe]])

    # ---- Step 4: small KD re-sweep on the best working small student ----
    w05_acc = top1(w05_base_run)
    if w05_acc is not None and w05_acc >= W05_OK_THRESHOLD:
        stu_tag, stu_args = f"w0.5_{best_recipe}", ["--width-mult", "0.5", *W05_RECIPES[best_recipe]]
        log(f"STEP 4/4: w0.5 fixed ({w05_acc:.2f} >= {W05_OK_THRESHOLD}) -> re-sweep on w0.5")
    else:
        stu_tag = "mbv3s"
        stu_args = ["--model", "mobilenetv3_small", *DEFAULT_RECIPE]
        log(f"STEP 4/4: w0.5 not fixed (baseline={w05_acc}) -> re-sweep on MobileNetV3-Small")
        train("student_mbv3s_cifar100_nols", "configs/student.yaml", stu_args)

    if teacher_ok:
        for T in SWEEP_T:
            for a in SWEEP_ALPHA:
                train(f"kd_cifar100_{stu_tag}_T{T}_a{a}_nols", "configs/kd.yaml",
                      [*stu_args, *kd_args(T, a)])
    else:
        log("re-sweep skipped (no teacher)")

    # ---- summary ----
    log(f"ALL DONE in {(time.time() - t_start) / 3600:.2f} h. Run status:")
    for s in STATUS:
        print("   " + s, flush=True)
    log("New results are in results/results.csv (run names ending in _nols) "
        "and results/quick_tests.csv.")


if __name__ == "__main__":
    main()
