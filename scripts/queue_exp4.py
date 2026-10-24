"""P25 Experiment 4: final KD runs on every student width (for the Pareto curve).

For each student (MBV2 w1.0, MBV2 w1.4, MobileNetV3-Small):
  1. 4-epoch quick recipe test: sgd0.01 vs adamw1e-3 (logged to results/quick_tests.csv)
  2. full 20-epoch baseline (no KD) with the winning recipe
  3. KD with the same recipe, T=2, alpha=0.9 (best setting from the w0.5 3-seed check)
MBV2 w0.5 is already done (student_mbv2_cifar100_w0.5_nols_adamw1e-3 / kd_..._T2.0_a0.9_nols).

All runs: label_smoothing=0, seed 42, teacher = teacher_resnet50_cifar100_nols.
Resumable: finished runs are skipped, interrupted runs resume from last.pt.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_exp4.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import queue_night3 as q  # noqa: E402  (reuses train(), kd_args(), top1(), log(), STATUS)

T, ALPHA = 2.0, 0.9
RECIPES = {k: q.W05_RECIPES[k] for k in ("sgd0.01", "adamw1e-3")}

# tag -> model args
STUDENTS = {
    "w1.0": ["--width-mult", "1.0"],
    "w1.4": ["--width-mult", "1.4"],
    "mbv3s": ["--model", "mobilenetv3_small"],
}

# Runs that already exist under an older name (same settings) -> reuse instead of retraining.
EXISTING_BASELINES = {("w1.0", "sgd0.01"): "student_mbv2_cifar100_w1.0_nols"}


def baseline_name(tag: str, recipe: str) -> str:
    if (tag, recipe) in EXISTING_BASELINES:
        return EXISTING_BASELINES[(tag, recipe)]
    prefix = "student_mbv3s_cifar100" if tag == "mbv3s" else f"student_mbv2_cifar100_{tag}"
    return f"{prefix}_nols_{recipe}"


def main() -> None:
    t_start = time.time()
    if not q.TEACHER_CKPT.exists():
        q.log(f"Teacher checkpoint missing: {q.TEACHER_CKPT}. Aborting.")
        sys.exit(1)

    winners: dict[str, str] = {}
    for tag, model_args in STUDENTS.items():
        q.log(f"===== student {tag} =====")
        # 1. quick recipe test
        for rname, recipe in RECIPES.items():
            q.train(f"qt_{tag}_{rname}", "configs/student.yaml",
                    [*model_args, "--epochs", str(q.QUICK_EPOCHS), *recipe],
                    csv_path=q.QUICK_CSV)
        scores = {r: q.top1(f"qt_{tag}_{r}", q.QUICK_CSV, "final_top1") for r in RECIPES}
        valid = {r: s for r, s in scores.items() if s is not None}
        best = max(valid, key=valid.get) if valid else "sgd0.01"
        winners[tag] = best
        q.log(f"{tag} quick test: " + ", ".join(f"{r}={s}" for r, s in scores.items())
              + f" -> using {best}")

        # 2. baseline, 3. KD (same recipe)
        stu_args = [*model_args, *RECIPES[best]]
        q.train(baseline_name(tag, best), "configs/student.yaml", stu_args)
        q.train(f"kd_cifar100_{tag}_{best}_T{T}_a{ALPHA}_nols", "configs/kd.yaml",
                [*stu_args, *q.kd_args(T, ALPHA)])

    # ---- summary: baseline vs KD for every width (incl. w0.5 from night 3) ----
    q.log(f"ALL DONE in {(time.time() - t_start) / 3600:.2f} h. Baseline vs KD (T={T}, a={ALPHA}):")
    rows = [("w0.5", "student_mbv2_cifar100_w0.5_nols_adamw1e-3",
             "kd_cifar100_w0.5_adamw1e-3_T2.0_a0.9_nols", "adamw1e-3")]
    for tag, best in winners.items():
        rows.append((tag, baseline_name(tag, best),
                     f"kd_cifar100_{tag}_{best}_T{T}_a{ALPHA}_nols", best))
    for tag, b, k, rec in rows:
        ba, ka = q.top1(b), q.top1(k)
        gain = f"{ka - ba:+.2f}" if ba is not None and ka is not None else "n/a"
        print(f"   {tag:6s} recipe={rec:10s} baseline={ba}  KD={ka}  gain={gain}", flush=True)
    q.log("Run status:")
    for s in q.STATUS:
        print("   " + s, flush=True)


if __name__ == "__main__":
    main()
