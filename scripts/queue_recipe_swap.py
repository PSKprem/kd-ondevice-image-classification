"""P25 optimizer de-confound: KD (T=2, alpha=0.9) with the OTHER recipe for every student.

Main results use each student's best recipe chosen on val (w0.5 / MBV3-S: AdamW, w1.0 / w1.4: SGD), so student size and
optimizer are mixed. All 8 baselines (4 students x 2 recipes) already exist in results/results_v2.csv, so only the 4 KD
runs with the non-chosen recipe are missing:  w0.5 SGD, MBV3-S SGD, w1.0 AdamW, w1.4 AdamW.
Afterwards KD gain is known for every student under BOTH optimizers -> results/recipe_swap_v2.csv.

Same seed 42, 20 epochs, label smoothing 0, same CIFAR-100 teacher, results in results/results_v2.csv.
Resumable: finished runs are skipped, interrupted runs continue from last.pt, a failed run is retried once.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_recipe_swap.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import queue_v2_night1 as q  # noqa: E402  (train(), RECIPES, STUDENTS, names, TEACHER_CKPT, metric, log)

OUT_CSV = ROOT / "results" / "recipe_swap_v2.csv"
ORDER = ["mbv3s", "w0.5", "w1.0", "w1.4"]  # cheapest first
T, ALPHA = 2.0, 0.9


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 recipe-swap queue")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()
    q.DRY_RUN = args.dry_run

    if not q.TEACHER_CKPT.exists():
        sys.exit(f"Teacher checkpoint missing: {q.TEACHER_CKPT}")
    ckpt = str(q.TEACHER_CKPT).replace("\\", "/")
    t0 = time.time()

    for tag in ORDER:
        for rec in q.RECIPES:
            if q.metric(q.baseline_name(tag, rec)) is None:
                sys.exit(f"Missing baseline {q.baseline_name(tag, rec)} - run queue_v2_night1.py first")
    chosen = {r["student"]: r["best_recipe"] for r in csv.DictReader(open(ROOT / "results" / "best_recipes_v2.csv"))}

    for tag in ORDER:
        other = [r for r in q.RECIPES if r != chosen[tag]][0]
        q.log(f"===== {tag}: chosen recipe {chosen[tag]}, running KD with the other recipe {other} =====")
        q.train(q.kd_name(tag, other, T, ALPHA), "configs/kd.yaml",
                [*q.STUDENTS[tag][0], *q.RECIPES[other], "--teacher-ckpt", ckpt,
                 "--temperature", str(T), "--alpha", str(ALPHA)])

    rows = []
    q.log("===== SUMMARY: KD gain per student and optimizer (seed 42) =====")
    for tag in ORDER:
        for rec in q.RECIPES:
            b_t, b_v = q.metric(q.baseline_name(tag, rec), "test_top1"), q.metric(q.baseline_name(tag, rec))
            k_t, k_v = q.metric(q.kd_name(tag, rec, T, ALPHA), "test_top1"), q.metric(q.kd_name(tag, rec, T, ALPHA))
            if None in (b_t, b_v, k_t, k_v):
                continue
            rows.append({"student": tag, "recipe": rec, "chosen_on_val": int(rec == chosen[tag]),
                         "baseline_test": f"{b_t:.2f}", "kd_test": f"{k_t:.2f}", "test_gain_pp": f"{k_t - b_t:+.2f}",
                         "baseline_val": f"{b_v:.2f}", "kd_val": f"{k_v:.2f}", "val_gain_pp": f"{k_v - b_v:+.2f}"})
            print("   " + "  ".join(f"{a}={v}" for a, v in rows[-1].items()), flush=True)
    if rows and not args.dry_run:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        q.log(f"wrote {OUT_CSV}")
    q.log(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h. Run status:")
    for s in q.STATUS:
        print("   " + s, flush=True)


if __name__ == "__main__":
    main()
