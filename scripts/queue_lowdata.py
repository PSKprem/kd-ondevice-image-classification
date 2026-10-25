"""P25 low-data study (v2 protocol): does KD help more when the student has less training data?

MBV2 w0.5 (AdamW) and w1.0 (SGD) trained on 10% and 25% of the CIFAR-100 train split (stratified, nested,
`--train-fraction`; val and test sets unchanged), baseline vs KD (T=2, alpha=0.9). 100% = the existing seed-42 runs.
Same 20 epochs for every fraction (so fewer optimizer steps with less data) and the SAME teacher, which was trained
on the full 45k images (usual KD setting; state this in the report).
Results: results/results_lowdata_v2.csv, summary results/summary_lowdata_v2.csv. Resumable like the other queues.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_lowdata.py [--dry-run]
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
import queue_v2_night1 as q  # noqa: E402  (train(), RECIPES, STUDENTS, names, TEACHER_CKPT, metric, log)

LOW_CSV = "results/results_lowdata_v2.csv"
SUMMARY_CSV = ROOT / "results" / "summary_lowdata_v2.csv"
FRACTIONS = [0.10, 0.25]
STUDENT_TAGS = ["w0.5", "w1.0"]
T, ALPHA = 2.0, 0.9


def run_name(kind: str, tag: str, rec: str, frac: float) -> str:
    pct = int(round(frac * 100))
    if kind == "base":
        return f"student_mbv2_cifar100_{tag}_v2_frac{pct}_{rec}"
    return f"kd_cifar100_{tag}_v2_frac{pct}_{rec}_T{T:g}_a{ALPHA:g}"


def metric_from(csv_path: str, name: str, key: str):
    """Value of `key` for run `name` in another results CSV (None if missing)."""
    from src.utils import read_results
    for r in read_results(str(ROOT / csv_path)):
        if r.get("run_name") == name:
            try:
                return float(r[key])
            except (TypeError, ValueError, KeyError):
                return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 low-data queue")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()
    q.DRY_RUN = args.dry_run
    q.RESULTS_CSV = LOW_CSV  # q.train() / q.row_for() read these module globals

    if not q.TEACHER_CKPT.exists():
        sys.exit(f"Teacher checkpoint missing: {q.TEACHER_CKPT}")
    recipes = json.loads(q.BEST_CFG_JSON.read_text(encoding="utf-8"))["recipes"]
    ckpt = str(q.TEACHER_CKPT).replace("\\", "/")
    t0 = time.time()

    for frac in FRACTIONS:
        for tag in STUDENT_TAGS:
            rec = recipes[tag]
            common = ["--train-fraction", str(frac), *q.STUDENTS[tag][0], *q.RECIPES[rec]]
            q.log(f"===== {tag} ({rec}) with {frac:.0%} of the training data =====")
            q.train(run_name("base", tag, rec, frac), "configs/student.yaml", common)
            q.train(run_name("kd", tag, rec, frac), "configs/kd.yaml",
                    [*common, "--teacher-ckpt", ckpt, "--temperature", str(T), "--alpha", str(ALPHA)])

    # full-data reference = existing seed-42 runs in results_v2.csv
    full = {}
    for tag in STUDENT_TAGS:
        rec = recipes[tag]
        full[tag] = (metric_from("results/results_v2.csv", q.baseline_name(tag, rec), "test_top1"),
                     metric_from("results/results_v2.csv", q.kd_name(tag, rec, T, ALPHA), "test_top1"))

    rows = []
    q.log("===== SUMMARY: KD gain vs amount of training data (test top-1) =====")
    for tag in STUDENT_TAGS:
        rec = recipes[tag]
        entries = []
        for frac in FRACTIONS:
            b, k = q.metric(run_name("base", tag, rec, frac), "test_top1"), q.metric(run_name("kd", tag, rec, frac), "test_top1")
            if b is not None and k is not None:
                entries.append((frac, b, k))
        if full[tag][0] is not None and full[tag][1] is not None:
            entries.append((1.0, full[tag][0], full[tag][1]))
        for frac, b, k in entries:
            rows.append({"student": tag, "recipe": rec, "train_fraction": frac, "baseline_test": f"{b:.2f}",
                         "kd_test": f"{k:.2f}", "kd_gain_pp": f"{k - b:+.2f}"})
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
