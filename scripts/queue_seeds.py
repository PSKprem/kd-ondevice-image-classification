"""P25 seed check: is the T=2 KD gain on MBV2 w0.5 (adamw recipe) real or seed noise?

Re-runs the w0.5 baseline and the two T=2 KD runs with extra seeds (seed 42 already
exists), then prints mean +- std per config over all available seeds.

Resumable like queue_night3: runs already in results.csv are skipped, interrupted
runs continue from last.pt.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_seeds.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import queue_night3 as q  # noqa: E402  (reuses train(), kd_args(), log(), STATUS)

EXTRA_SEEDS = [0, 1]
W05 = ["--width-mult", "0.5", *q.W05_RECIPES["adamw1e-3"]]

# config label -> (seed-42 run name, config file, extra args)
CONFIGS = {
    "baseline": ("student_mbv2_cifar100_w0.5_nols_adamw1e-3", "configs/student.yaml", W05),
    "KD T2 a0.5": ("kd_cifar100_w0.5_adamw1e-3_T2.0_a0.5_nols", "configs/kd.yaml",
                   [*W05, *q.kd_args(2.0, 0.5)]),
    "KD T2 a0.9": ("kd_cifar100_w0.5_adamw1e-3_T2.0_a0.9_nols", "configs/kd.yaml",
                   [*W05, *q.kd_args(2.0, 0.9)]),
}


def main() -> None:
    for seed in EXTRA_SEEDS:
        q.log(f"=== seed {seed} ===")
        for base_name, config, extra in CONFIGS.values():
            q.train(f"{base_name}_s{seed}", config, [*extra, "--seed", str(seed)])

    q.log("Summary (best_top1 over seeds 42 + extra):")
    base_mean = None
    for label, (base_name, _, _) in CONFIGS.items():
        names = [base_name] + [f"{base_name}_s{s}" for s in EXTRA_SEEDS]
        accs = [a for a in (q.top1(n) for n in names) if a is not None]
        if not accs:
            print(f"   {label:12s} no results", flush=True)
            continue
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        if label == "baseline":
            base_mean = mean
        delta = f"  delta={mean - base_mean:+.2f}" if base_mean is not None and label != "baseline" else ""
        print(f"   {label:12s} n={len(accs)}  mean={mean:.2f}  std={std:.2f}  "
              f"runs={[round(a, 2) for a in accs]}{delta}", flush=True)
    q.log("Run status:")
    for s in q.STATUS:
        print("   " + s, flush=True)


if __name__ == "__main__":
    main()
