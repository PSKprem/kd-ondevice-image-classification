"""P25 from-scratch comparison (v2 protocol): MBV2 w0.5 / w1.0 trained WITHOUT ImageNet weights, baseline vs KD.

Question: does KD help more when the student has no pretrained weights?
Same recipe per student as the pretrained runs (w0.5 AdamW, w1.0 SGD), 20 epochs, KD T=2 alpha=0.9, same
CIFAR-100 teacher. Results: results/results_scratch_v2.csv (separate file), summary results/summary_scratch_v2.csv.
Resumable: finished runs are skipped, interrupted runs continue from last.pt, a failed run is retried once.
Reuses train() from queue_v2_night1.py.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_scratch.py [--dry-run] [--epochs 20]
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
import queue_v2_night1 as q  # noqa: E402  (train(), RECIPES, STUDENTS, TEACHER_CKPT, row_for, metric, log)

SCRATCH_CSV = "results/results_scratch_v2.csv"
SUMMARY_CSV = ROOT / "results" / "summary_scratch_v2.csv"
STUDENT_TAGS = ["w0.5", "w1.0"]
T, ALPHA = 2.0, 0.9
NO_PRETRAINED = ["--no-pretrained"]


def baseline_name(tag: str, recipe: str) -> str:
    return f"student_mbv2_cifar100_{tag}_v2_scratch_{recipe}"


def kd_name(tag: str, recipe: str) -> str:
    return f"kd_cifar100_{tag}_v2_scratch_{recipe}_T{T:g}_a{ALPHA:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 from-scratch queue")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    q.DRY_RUN = args.dry_run
    q.RESULTS_CSV = SCRATCH_CSV  # q.train() / q.row_for() read these module globals
    q.EPOCHS = args.epochs

    if not q.TEACHER_CKPT.exists():
        sys.exit(f"Teacher checkpoint missing: {q.TEACHER_CKPT}")
    recipes = json.loads(q.BEST_CFG_JSON.read_text(encoding="utf-8"))["recipes"]
    ckpt = str(q.TEACHER_CKPT).replace("\\", "/")
    t0 = time.time()
    q.log(f"From-scratch queue: {args.epochs} epochs, KD T={T:g} alpha={ALPHA:g}")

    for tag in STUDENT_TAGS:
        rec = recipes[tag]
        model_args = q.STUDENTS[tag][0]
        q.log(f"===== student {tag} from scratch ({rec}) =====")
        q.train(baseline_name(tag, rec), "configs/student.yaml", [*NO_PRETRAINED, *model_args, *q.RECIPES[rec]])
        q.train(kd_name(tag, rec), "configs/kd.yaml",
                [*NO_PRETRAINED, *model_args, *q.RECIPES[rec], "--teacher-ckpt", ckpt,
                 "--temperature", str(T), "--alpha", str(ALPHA)])

    rows = []
    q.log("===== SUMMARY (CIFAR-100, from scratch, test top-1) =====")
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
