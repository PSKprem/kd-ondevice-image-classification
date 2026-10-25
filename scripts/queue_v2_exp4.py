"""P25 v2, Experiment 4 = final KD runs on every student + seed check (val-selected, test reported).

  Part A: KD with the best (T, alpha) from results/best_config_v2.json on every student,
          each with ITS OWN best baseline recipe (results/best_config_v2.json -> "recipes").
          w1.0 already exists from the sweep (same run name) and is skipped.
  Part B: seed check on w0.5 and w1.0: baseline + KD re-run with seeds 0 and 1
          (seed 42 = the runs above). Tells us whether the KD gain is larger than seed noise.
          The train/val split is fixed (split_seed 42), only the training seed changes.
  Summary: results/kd_gain_v2.csv  (test_top1 mean +- std per student, baseline vs KD, gain).

Reuses train() from queue_v2_night1.py: same teacher, 20 epochs, label smoothing 0,
logs to results/results_v2.csv, checkpoints in D:/kd-project/runs_v2.
Resumable: finished runs are skipped, interrupted runs continue from last.pt.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_v2_exp4.py [--dry-run] [--seeds]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import queue_v2_night1 as q  # noqa: E402  (train(), STUDENTS, RECIPES, names, row_for, metric, log)

GAIN_CSV = ROOT / "results" / "kd_gain_v2.csv"
MAIN_SEED = 42
EXTRA_SEEDS = [0, 1]
SEED_STUDENTS = ["w0.5", "w1.0"]
ALL_STUDENTS = ["w0.5", "w1.0", "w1.4", "mbv3s"]
# One common KD setting for all students (simple + defensible: within ~0.4 pp of the w1.0 grid best,
# no harm on w0.5, and it gave +1.7..+2.0 pp on w1.4 / MBV3-S in v1).
COMMON_T, COMMON_ALPHA = 2.0, 0.9


def with_seed(name: str, seed: int) -> str:
    return name if seed == MAIN_SEED else f"{name}_s{seed}"


def baseline_args(tag: str, recipe: str, seed: int) -> list[str]:
    extra = [*q.STUDENTS[tag][0], *q.RECIPES[recipe]]
    return extra if seed == MAIN_SEED else [*extra, "--seed", str(seed)]


def kd_args(tag: str, recipe: str, T: float, a: float, seed: int) -> list[str]:
    extra = [*q.STUDENTS[tag][0], *q.RECIPES[recipe],
             "--teacher-ckpt", str(q.TEACHER_CKPT).replace("\\", "/"),
             "--temperature", str(T), "--alpha", str(a)]
    return extra if seed == MAIN_SEED else [*extra, "--seed", str(seed)]


def run_pair(tag: str, recipe: str, T: float, a: float, seed: int) -> None:
    """Baseline + KD for one student/seed (baseline is normally already done for seed 42)."""
    q.train(with_seed(q.baseline_name(tag, recipe), seed), "configs/student.yaml",
            baseline_args(tag, recipe, seed))
    q.train(with_seed(q.kd_name(tag, recipe, T, a), seed), "configs/kd.yaml",
            kd_args(tag, recipe, T, a, seed))


def mean_std(vals: list[float]) -> tuple[float, float]:
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def summarize(recipes: dict[str, str], T: float, a: float) -> None:
    q.log("===== SUMMARY: baseline vs KD (test_top1, mean +- std over available seeds) =====")
    rows = []
    for tag in ALL_STUDENTS:
        if tag not in recipes:
            continue
        rec = recipes[tag]
        base, kd = [], []
        for seed in [MAIN_SEED, *EXTRA_SEEDS]:
            b = q.metric(with_seed(q.baseline_name(tag, rec), seed), "test_top1")
            k = q.metric(with_seed(q.kd_name(tag, rec, T, a), seed), "test_top1")
            if b is not None and k is not None:  # only seeds where both finished
                base.append(b)
                kd.append(k)
        if not base:
            continue
        (bm, bs), (km, ks) = mean_std(base), mean_std(kd)
        params = q.metric(q.baseline_name(tag, rec), "params_m")
        rows.append({
            "student": tag, "recipe": rec, "params_m": f"{params:.3f}" if params else "",
            "n_seeds": len(base),
            "baseline_test_mean": f"{bm:.2f}", "baseline_test_std": f"{bs:.2f}",
            "kd_test_mean": f"{km:.2f}", "kd_test_std": f"{ks:.2f}",
            "kd_gain_pp": f"{km - bm:+.2f}", "T": T, "alpha": a,
        })
    if not rows:
        q.log("Nothing to summarize yet.")
        return
    for r in rows:
        print("   " + "  ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    if not q.DRY_RUN:
        with open(GAIN_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        q.log(f"wrote {GAIN_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 v2 Experiment 4 queue")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--seeds", action="store_true", help="also run Part B (extra seeds 0,1), optional")
    args = ap.parse_args()
    q.DRY_RUN = args.dry_run

    if not q.BEST_CFG_JSON.exists():
        sys.exit(f"Missing {q.BEST_CFG_JSON} - run scripts/queue_v2_night1.py first.")
    cfg = json.loads(q.BEST_CFG_JSON.read_text(encoding="utf-8"))
    recipes, T, a = cfg["recipes"], COMMON_T, COMMON_ALPHA
    if not q.TEACHER_CKPT.exists():
        sys.exit(f"Teacher checkpoint missing: {q.TEACHER_CKPT}")
    q.log(f"Using common (T, alpha) = ({T:g}, {a:g}); recipes = {recipes}")

    t0 = time.time()
    q.log("===== PART A: KD with best (T, alpha) on every student, seed 42 =====")
    for tag in ALL_STUDENTS:
        if tag in recipes:
            run_pair(tag, recipes[tag], T, a, MAIN_SEED)

    if args.seeds:
        q.log(f"===== PART B: seed check {EXTRA_SEEDS} on {SEED_STUDENTS} =====")
        for seed in EXTRA_SEEDS:
            for tag in SEED_STUDENTS:
                if tag in recipes:
                    run_pair(tag, recipes[tag], T, a, seed)

    summarize(recipes, T, a)
    q.log(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h. Run status:")
    for s in q.STATUS:
        print("   " + s, flush=True)


if __name__ == "__main__":
    main()
