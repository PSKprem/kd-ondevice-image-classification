"""P25 — on-device latency of the exported TFLite models via adb + TFLite `benchmark_model` (CPU, XNNPACK).

Everything lives in one folder on the phone (/data/local/tmp/p25); nothing is installed, only that folder is touched.
Order: repetitions outermost, models innermost, so thermal drift is spread over all models instead of hitting the last one.
The script pauses when the battery temperature is above --max-temp. Rows are appended to results/phone_latency_v2.csv
(resumable: finished (file, threads, rep) combinations are skipped).

Usage (from repo root, any python):
    python scripts/phone_benchmark.py --adb "D:/Downloads/platform-tools-latest-windows/platform-tools/adb.exe" [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phone_latency_v2.csv"
REMOTE = "/data/local/tmp/p25"
MODELS = []  # (file on phone, student, precision)
for _tag in ("w0.5", "mbv3s", "w1.0", "w1.4"):
    for _suffix, _prec in (("fp32", "FP32"), ("int8dyn", "INT8 dynamic"), ("int8mixed", "INT8 mixed")):
        MODELS.append((f"{_tag}_kd_{_suffix}.tflite", _tag, _prec))
MODELS.append(("teacher_fp32.tflite", "teacher", "FP32"))
FIELDS = ["file", "student", "precision", "threads", "rep", "avg_ms", "first_ms", "init_ms", "mem_mb",
          "temp_before_c", "temp_after_c", "num_runs"]


def sh(adb: str, cmd: str, timeout: int = 600) -> str:
    r = subprocess.run([adb, "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or "")


def battery_temp_c(adb: str) -> float:
    m = re.search(r"temperature:\s*(\d+)", sh(adb, "dumpsys battery"))
    return int(m.group(1)) / 10.0 if m else float("nan")


def done_keys() -> set:
    if not OUT.exists():
        return set()
    with open(OUT, encoding="utf-8") as f:
        return {(r["file"], r["threads"], r["rep"]) for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 phone latency benchmark")
    ap.add_argument("--adb", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-temp", type=float, default=40.0, help="pause while battery temperature is above this (C)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    finished = done_keys()
    new = not OUT.exists()
    t0 = time.time()
    f = None if args.dry_run else open(OUT, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDS) if f else None
    if writer and new:
        writer.writeheader()
    try:
        for rep in range(1, args.reps + 1):
            for threads in args.threads:
                for fname, student, prec in MODELS:
                    if (fname, str(threads), str(rep)) in finished:
                        continue
                    cmd = (f"{REMOTE}/benchmark_model --graph={REMOTE}/{fname} --num_threads={threads} "
                           f"--num_runs={args.runs} --warmup_runs={args.warmup}")
                    if args.dry_run:
                        print(cmd)
                        continue
                    waited = 0
                    while battery_temp_c(args.adb) > args.max_temp and waited < 600:
                        print(f"   phone is hot ({battery_temp_c(args.adb)} C) - cooling 30 s", flush=True)
                        time.sleep(30)
                        waited += 30
                    tb = battery_temp_c(args.adb)
                    out = sh(args.adb, cmd)
                    m = re.search(r"Init:\s*(\d+),\s*First inference:\s*(\d+),\s*Warmup \(avg\):\s*([\d.]+),\s*Inference \(avg\):\s*([\d.]+)", out)
                    mem = re.search(r"overall=([\d.]+)", out)
                    if not m:
                        print(f"   FAILED {fname} threads={threads}: {out[-300:]}", flush=True)
                        continue
                    row = {"file": fname, "student": student, "precision": prec, "threads": threads, "rep": rep,
                           "avg_ms": f"{float(m.group(4)) / 1000:.2f}", "first_ms": f"{float(m.group(2)) / 1000:.2f}",
                           "init_ms": f"{float(m.group(1)) / 1000:.2f}", "mem_mb": mem.group(1) if mem else "",
                           "temp_before_c": tb, "temp_after_c": battery_temp_c(args.adb), "num_runs": args.runs}
                    writer.writerow(row)
                    f.flush()
                    print(f"rep{rep} t={threads} {fname:26s} avg={row['avg_ms']:>8s} ms  temp {tb}->{row['temp_after_c']}", flush=True)
    finally:
        if f:
            f.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
