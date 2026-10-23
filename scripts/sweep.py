"""P25 — T x alpha sweep driver.

Runs the KD grid (T in {1,2,4,8} x alpha in {0.1,0.5,0.9}) on one student width,
skipping any (T, alpha) already present in results/results.csv so the sweep is
resumable. Each run is launched as a subprocess calling src.train.

Usage:
    python scripts/sweep.py --config configs/kd.yaml \
        --teacher-ckpt D:/kd-project/runs/teacher_resnet50_cifar100/best.pt \
        --width-mult 1.0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# make `src` importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, read_results  # noqa: E402

TEMPERATURES = [1.0, 2.0, 4.0, 8.0]
ALPHAS = [0.1, 0.5, 0.9]


def already_done(results, dataset, width, T, alpha) -> bool:
    for r in results:
        try:
            if (r.get("dataset") == dataset
                    and abs(float(r.get("width_mult", -1)) - width) < 1e-6
                    and int(float(r.get("kd", 0))) == 1
                    and abs(float(r.get("temperature", -1)) - T) < 1e-6
                    and abs(float(r.get("alpha", -1)) - alpha) < 1e-6):
                return True
        except (TypeError, ValueError):
            continue
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 T x alpha sweep")
    ap.add_argument("--config", default="configs/kd.yaml")
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_csv = cfg["results_csv"]
    dataset = cfg["dataset"]

    planned, skipped = 0, 0
    for T in TEMPERATURES:
        for alpha in ALPHAS:
            results = read_results(results_csv)  # re-read so finished rows count
            if already_done(results, dataset, args.width_mult, T, alpha):
                skipped += 1
                print(f"[sweep] skip T={T} alpha={alpha} (already in {results_csv})")
                continue
            run_name = f"kd_{dataset}_w{args.width_mult}_T{T}_a{alpha}"
            cmd = [
                sys.executable, "-m", "src.train",
                "--config", args.config,
                "--teacher-ckpt", args.teacher_ckpt,
                "--width-mult", str(args.width_mult),
                "--temperature", str(T),
                "--alpha", str(alpha),
                "--run-name", run_name,
            ]
            planned += 1
            print(f"[sweep] RUN {run_name}: {' '.join(cmd)}")
            if args.dry_run:
                continue
            subprocess.run(cmd, check=True)

    print(f"[sweep] finished. launched={planned} skipped={skipped}")


if __name__ == "__main__":
    main()
