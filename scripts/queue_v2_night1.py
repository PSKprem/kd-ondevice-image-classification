"""P25 v2, Night 1 queue = Task 3 + 4 + 5 (all selection on the VAL split, never on test).

  Task 3: ResNet-50 teacher on the 45k train split, label smoothing 0.
  Task 4: 4 students (MBV2 w0.5 / w1.0 / w1.4, MBV3-Small) x 2 recipes
          (sgd0.01, adamw1e-3), full 20 epochs = 8 baseline runs (no KD).
          Per student the recipe with the higher best_val_top1 wins
          -> results/best_recipes_v2.csv
  Task 5: full KD T x alpha grid on w1.0 with ITS best recipe:
          T in {1,2,4,8} x alpha in {0.1,0.5,0.9} = 12 runs.
          Best (T, alpha) = highest best_val_top1 -> results/best_config_v2.json
          + heatmap results/figures/heatmap_v2.png

Everything logs to results/results_v2.csv (checkpoints in D:/kd-project/runs_v2).
Resumable: runs already in the CSV are skipped, interrupted runs continue from last.pt,
a failed run is retried once. Just start the script again after any interruption.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u scripts/queue_v2_night1.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import read_results  # noqa: E402

RESULTS_CSV = "results/results_v2.csv"
RECIPE_TABLE = ROOT / "results" / "best_recipes_v2.csv"
BEST_CFG_JSON = ROOT / "results" / "best_config_v2.json"
HEATMAP_PNG = ROOT / "results" / "figures" / "heatmap_v2.png"

RUNS_DIR = Path("D:/kd-project/runs_v2")
TEACHER_RUN = "teacher_resnet50_cifar100_v2"
TEACHER_CKPT = RUNS_DIR / TEACHER_RUN / "best.pt"

EPOCHS = 20
SWEEP_STUDENT = "w1.0"
TEMPERATURES = [1.0, 2.0, 4.0, 8.0]
ALPHAS = [0.1, 0.5, 0.9]

# Same two recipes as the v1 experiments (label smoothing 0, warmup 1 from the config).
RECIPES = {
    "sgd0.01": ["--optimizer", "sgd", "--lr", "0.01", "--weight-decay", "0.0005"],
    "adamw1e-3": ["--optimizer", "adamw", "--lr", "0.001", "--weight-decay", "0.05"],
}
# tag -> (model args, baseline run-name prefix)
STUDENTS = {
    "w0.5": (["--width-mult", "0.5"], "student_mbv2_cifar100_w0.5_v2"),
    "w1.0": (["--width-mult", "1.0"], "student_mbv2_cifar100_w1.0_v2"),
    "w1.4": (["--width-mult", "1.4"], "student_mbv2_cifar100_w1.4_v2"),
    "mbv3s": (["--model", "mobilenetv3_small"], "student_mbv3s_cifar100_v2"),
}

DRY_RUN = False
STATUS: list[str] = []


def log(msg: str) -> None:
    print(f"[queue {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def row_for(run_name: str) -> dict | None:
    path = ROOT / RESULTS_CSV
    if not path.exists():
        return None
    for r in read_results(str(path)):
        if r.get("run_name") == run_name:
            return r
    return None


def metric(run_name: str, key: str = "best_val_top1") -> float | None:
    r = row_for(run_name)
    try:
        return float(r[key]) if r else None
    except (TypeError, ValueError, KeyError):
        return None


def train(run_name: str, config: str, extra: list[str]) -> bool:
    """Run one training job unless already logged. One retry (resumes from last.pt)."""
    if row_for(run_name) is not None:
        log(f"SKIP {run_name} (already in {RESULTS_CSV})")
        STATUS.append(f"skip  {run_name}")
        return True
    cmd = [sys.executable, "-u", "-m", "src.train", "--config", config,
           "--run-name", run_name, "--label-smoothing", "0", "--epochs", str(EPOCHS),
           "--results-csv", RESULTS_CSV, "--resume", *extra]
    log(f"RUN  {run_name}\n      {' '.join(cmd)}")
    if DRY_RUN:
        return True
    t0 = time.time()
    ok = False
    for attempt in (1, 2):
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        ok = rc == 0 and row_for(run_name) is not None
        if ok:
            break
        log(f"attempt {attempt} failed (rc={rc}) for {run_name}" + (" -> retrying" if attempt == 1 else ""))
    mins = (time.time() - t0) / 60
    STATUS.append(f"{'ok  ' if ok else 'FAIL'}  {run_name}  ({mins:.1f} min)")
    log(f"{'DONE' if ok else 'FAILED'} {run_name} in {mins:.1f} min")
    return ok


def baseline_name(tag: str, recipe: str) -> str:
    return f"{STUDENTS[tag][1]}_{recipe}"


def kd_name(tag: str, recipe: str, T: float, a: float) -> str:
    return f"kd_cifar100_{tag}_v2_{recipe}_T{T:g}_a{a:g}"


# --------------------------------------------------------------------------- #
def task3_teacher() -> bool:
    log("===== TASK 3: teacher ResNet-50, 45k train, label smoothing 0 =====")
    train(TEACHER_RUN, "configs/teacher.yaml", RECIPES["sgd0.01"])
    r = row_for(TEACHER_RUN)
    if r is None or not TEACHER_CKPT.exists():
        log("Teacher missing -> cannot continue.")
        return False
    v, t = float(r["best_val_top1"]), float(r["test_top1"])
    log(f"TEACHER  best_val_top1={v:.2f}  test_top1={t:.2f}  final_test_top1={r['final_test_top1']}")
    if abs(v - t) > 3.0:
        log(f"WARNING: val/test gap {abs(v - t):.2f} pp is large - check the split before trusting results")
    return True


def task4_baselines() -> dict[str, str]:
    """Run 8 baselines, pick the best recipe per student by VAL. Returns tag -> recipe."""
    log("===== TASK 4: 4 students x 2 recipes, 20 epochs, no KD =====")
    for tag, (model_args, _) in STUDENTS.items():
        for rname, recipe in RECIPES.items():
            train(baseline_name(tag, rname), "configs/student.yaml", [*model_args, *recipe])

    best: dict[str, str] = {}
    rows = []
    for tag in STUDENTS:
        vals = {r: metric(baseline_name(tag, r)) for r in RECIPES}
        valid = {r: v for r, v in vals.items() if v is not None}
        if not valid:
            log(f"{tag}: no finished baseline -> excluded")
            continue
        win = max(valid, key=valid.get)  # ties -> first recipe listed
        best[tag] = win
        rows.append({
            "student": tag,
            **{f"val_{r}": (f"{vals[r]:.2f}" if vals[r] is not None else "") for r in RECIPES},
            "best_recipe": win, "best_val_top1": f"{valid[win]:.2f}",
            "test_top1_of_best": f"{metric(baseline_name(tag, win), 'test_top1'):.2f}",
            "run_name": baseline_name(tag, win),
        })
    if rows and not DRY_RUN:
        RECIPE_TABLE.parent.mkdir(parents=True, exist_ok=True)
        with open(RECIPE_TABLE, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"wrote {RECIPE_TABLE}")
    log("Best recipe per student (chosen on VAL):")
    for r in rows:
        print("   " + "  ".join(f"{k}={v}" for k, v in r.items() if k != "run_name"), flush=True)
    return best


def save_heatmap(grid: dict[tuple[float, float], float], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [[grid.get((T, a), float("nan")) for a in ALPHAS] for T in TEMPERATURES]
    plt.figure(figsize=(5, 4))
    im = plt.imshow(data, aspect="auto", origin="lower")
    plt.colorbar(im, label="best val top-1 (%)")
    for i, T in enumerate(TEMPERATURES):
        for j, a in enumerate(ALPHAS):
            if data[i][j] == data[i][j]:
                plt.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", fontsize=8, color="w")
    plt.xticks(range(len(ALPHAS)), [f"{a:g}" for a in ALPHAS])
    plt.yticks(range(len(TEMPERATURES)), [f"{t:g}" for t in TEMPERATURES])
    plt.xlabel("alpha")
    plt.ylabel("temperature T")
    plt.title(title)
    plt.tight_layout()
    HEATMAP_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(HEATMAP_PNG, dpi=150)
    plt.close()
    log(f"wrote {HEATMAP_PNG}")


def task5_sweep(best_recipe: dict[str, str]) -> None:
    log(f"===== TASK 5: full T x alpha sweep on {SWEEP_STUDENT} (12 KD runs) =====")
    if SWEEP_STUDENT not in best_recipe:
        log(f"No baseline for {SWEEP_STUDENT} -> sweep skipped.")
        return
    recipe = best_recipe[SWEEP_STUDENT]
    model_args = STUDENTS[SWEEP_STUDENT][0]
    log(f"{SWEEP_STUDENT} best recipe = {recipe}")
    for T in TEMPERATURES:
        for a in ALPHAS:
            train(kd_name(SWEEP_STUDENT, recipe, T, a), "configs/kd.yaml",
                  [*model_args, *RECIPES[recipe],
                   "--teacher-ckpt", str(TEACHER_CKPT).replace("\\", "/"),
                   "--temperature", str(T), "--alpha", str(a)])

    grid = {}
    for T in TEMPERATURES:
        for a in ALPHAS:
            v = metric(kd_name(SWEEP_STUDENT, recipe, T, a))
            if v is not None:
                grid[(T, a)] = v
    if not grid:
        log("No finished sweep runs.")
        return
    (bT, bA), bV = max(grid.items(), key=lambda kv: kv[1])
    log(f"BEST (T, alpha) on VAL = ({bT:g}, {bA:g})  best_val_top1={bV:.2f}  "
        f"[{len(grid)}/{len(TEMPERATURES) * len(ALPHAS)} runs finished]")
    if DRY_RUN:
        return
    save_heatmap(grid, f"KD on MBV2 {SWEEP_STUDENT} ({recipe}) - val top-1")
    BEST_CFG_JSON.write_text(json.dumps({
        "selection": "best_val_top1 (val split, never test)",
        "recipes": best_recipe,
        "sweep_student": SWEEP_STUDENT,
        "sweep_runs_finished": len(grid),
        "best_T": bT, "best_alpha": bA, "best_val_top1": bV,
        "val_grid": {f"T{T:g}_a{a:g}": v for (T, a), v in sorted(grid.items())},
    }, indent=2), encoding="utf-8")
    log(f"wrote {BEST_CFG_JSON}")


def main() -> None:
    global DRY_RUN
    ap = argparse.ArgumentParser(description="P25 v2 night-1 queue (Tasks 3-5)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    DRY_RUN = ap.parse_args().dry_run

    t0 = time.time()
    if not task3_teacher() and not DRY_RUN:
        sys.exit(1)
    best = task4_baselines()
    task5_sweep(best)

    log(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h. Run status:")
    for s in STATUS:
        print("   " + s, flush=True)
    log("Results: results/results_v2.csv, results/best_recipes_v2.csv, "
        "results/best_config_v2.json, results/figures/heatmap_v2.png")


if __name__ == "__main__":
    main()
