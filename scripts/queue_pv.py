"""P25 PlantVillage study (v2 protocol, block split): teacher + MBV2 w0.5 / w1.0, baseline and KD = 5 runs.

Same recipes as CIFAR-100 (NOT re-tuned on PlantVillage - state this in the report):
  teacher ResNet-50: SGD 0.01 · w0.5: AdamW 1e-3 · w1.0: SGD 0.01 · KD: T=2, alpha=0.9, label smoothing 0.
Results go to results/results_pv_v2.csv (separate from CIFAR-100 so plot_results.py is not confused);
checkpoints in D:/kd-project/runs_v2. Resumable: finished runs are skipped, interrupted runs continue from
last.pt, a failed run is retried once. Reuses train() from queue_v2_night1.py.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_pv.py [--dry-run] [--epochs 20]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import queue_v2_night1 as q  # noqa: E402  (train(), RECIPES, STUDENTS, row_for, metric, log)

PV_CSV = "results/results_pv_v2.csv"
SUMMARY_CSV = ROOT / "results" / "summary_pv_v2.csv"
TEACHER_RUN = "teacher_resnet50_pv_v2"
TEACHER_CKPT = q.RUNS_DIR / TEACHER_RUN / "best.pt"
STUDENT_TAGS = ["w0.5", "w1.0"]
T, ALPHA = 2.0, 0.9  # same common KD setting as CIFAR-100
PV = ["--dataset", "plantvillage"]


def baseline_name(tag: str, recipe: str) -> str:
    return f"student_mbv2_pv_{tag}_v2_{recipe}"


def kd_name(tag: str, recipe: str) -> str:
    return f"kd_pv_{tag}_v2_{recipe}_T{T:g}_a{ALPHA:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 PlantVillage queue")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--epochs", type=int, default=20, help="epochs for every run (default 20, as CIFAR-100)")
    args = ap.parse_args()
    q.DRY_RUN = args.dry_run
    q.RESULTS_CSV = PV_CSV  # q.train() / q.row_for() read these module globals
    q.EPOCHS = args.epochs

    recipes = json.loads(q.BEST_CFG_JSON.read_text(encoding="utf-8"))["recipes"]
    t0 = time.time()
    q.log(f"PlantVillage queue: {args.epochs} epochs, KD T={T:g} alpha={ALPHA:g}, recipes "
          f"{ {t: recipes[t] for t in STUDENT_TAGS} }")

    q.log("===== teacher ResNet-50 =====")
    q.train(TEACHER_RUN, "configs/teacher.yaml", [*PV, *q.RECIPES["sgd0.01"]])
    if not args.dry_run and not TEACHER_CKPT.exists():
        sys.exit(f"Teacher checkpoint missing: {TEACHER_CKPT}")
    ckpt = str(TEACHER_CKPT).replace("\\", "/")

    for tag in STUDENT_TAGS:
        rec = recipes[tag]
        model_args = q.STUDENTS[tag][0]
        q.log(f"===== student {tag} ({rec}) =====")
        q.train(baseline_name(tag, rec), "configs/student.yaml", [*PV, *model_args, *q.RECIPES[rec]])
        q.train(kd_name(tag, rec), "configs/kd.yaml",
                [*PV, *model_args, *q.RECIPES[rec], "--teacher-ckpt", ckpt,
                 "--temperature", str(T), "--alpha", str(ALPHA)])

    rows = []
    q.log("===== SUMMARY (PlantVillage, test top-1) =====")
    teacher_test = q.metric(TEACHER_RUN, "test_top1")
    if teacher_test is not None:
        print(f"   teacher ResNet-50  test={teacher_test:.2f}  val={q.metric(TEACHER_RUN):.2f}", flush=True)
    for tag in STUDENT_TAGS:
        rec = recipes[tag]
        b, k = q.metric(baseline_name(tag, rec), "test_top1"), q.metric(kd_name(tag, rec), "test_top1")
        if b is None or k is None:
            continue
        rows.append({"student": tag, "recipe": rec, "baseline_test": f"{b:.2f}", "kd_test": f"{k:.2f}",
                     "kd_gain_pp": f"{k - b:+.2f}",
                     "baseline_val": f"{q.metric(baseline_name(tag, rec)):.2f}",
                     "kd_val": f"{q.metric(kd_name(tag, rec)):.2f}", "epochs": args.epochs})
        print("   " + "  ".join(f"{a}={v}" for a, v in rows[-1].items()), flush=True)
    if rows and not args.dry_run:
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        q.log(f"wrote {SUMMARY_CSV}")
    q.log(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h. Run status:")
    for s in q.STATUS:
        print("   " + s, flush=True)


if __name__ == "__main__":
    main()
