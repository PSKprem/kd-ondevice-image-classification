"""P25 — plots from results/results.csv.

Produces (into results/figures/):
  - pareto.png        accuracy vs params (baseline vs KD)
  - heatmap.png       T x alpha -> best_top1 (one width)
  - kd_gain.png       KD top1 - baseline top1 vs width

Usage:
    python scripts/plot_results.py --csv results/results.csv --out results/figures
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import read_results  # noqa: E402


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _pretrained(row) -> bool:
    """Main results use pretrained (transfer-learning) runs only."""
    return int(_f(row, "pretrained", 1)) == 1


# ---- Final (Experiment 4) run set: no label smoothing, adamw, seed 42, KD T=2 a=0.9 ----
FINAL_STUDENTS = [  # (label, baseline run, KD run)
    ("MBV2 w0.5", "student_mbv2_cifar100_w0.5_nols_adamw1e-3",
     "kd_cifar100_w0.5_adamw1e-3_T2.0_a0.9_nols"),
    ("MBV3-Small", "student_mbv3s_cifar100_nols_adamw1e-3",
     "kd_cifar100_mbv3s_adamw1e-3_T2.0_a0.9_nols"),
    ("MBV2 w1.0", "student_mbv2_cifar100_w1.0_nols_adamw1e-3",
     "kd_cifar100_w1.0_adamw1e-3_T2.0_a0.9_nols"),
    ("MBV2 w1.4", "student_mbv2_cifar100_w1.4_nols_adamw1e-3",
     "kd_cifar100_w1.4_adamw1e-3_T2.0_a0.9_nols"),
]
FINAL_TEACHER = "teacher_resnet50_cifar100_nols"


def final_pairs(rows):
    """Return [(label, params_m, base_top1, kd_top1)] for students with both runs present."""
    by_name = {r.get("run_name"): r for r in rows}
    pairs = []
    for label, b, k in FINAL_STUDENTS:
        if b in by_name and k in by_name:
            pairs.append((label, _f(by_name[b], "params_m"),
                          _f(by_name[b], "best_top1"), _f(by_name[k], "best_top1")))
    return sorted(pairs, key=lambda p: p[1]), by_name.get(FINAL_TEACHER)


def plot_pareto(rows, out: Path) -> None:
    pairs, teacher = final_pairs(rows)
    if not pairs:
        print("[plot] no final baseline/KD pairs; skipping pareto.")
        return
    xs = [p[1] for p in pairs]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [p[2] for p in pairs], "o--", label="baseline (no KD)")
    plt.plot(xs, [p[3] for p in pairs], "^-", label="KD (T=2, α=0.9)")
    for label, x, b, k in pairs:
        plt.annotate(f"{label}\n+{k - b:.2f}", (x, k), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    if teacher:
        # teacher drawn as reference line (23.7M params would squash the x-axis)
        ty = _f(teacher, "best_top1")
        plt.axhline(ty, color="gray", ls=":", lw=1)
        plt.text(0.01, ty + 0.15, f"teacher ResNet-50 ({_f(teacher, 'params_m'):.1f}M): {ty:.2f}",
                 transform=plt.gca().get_yaxis_transform(), ha="left", fontsize=8, color="gray")
    plt.xlabel("student params (M)")
    plt.ylabel("CIFAR-100 top-1 (%)")
    plt.title("Accuracy vs model size: baseline vs KD")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "pareto.png", dpi=150)
    plt.close()
    print(f"[plot] wrote {out / 'pareto.png'}")


def plot_heatmap(rows, out: Path, pattern: str, fname: str, title: str) -> None:
    rx = re.compile(pattern)
    kd = [r for r in rows if int(_f(r, "kd")) == 1 and rx.fullmatch(r.get("run_name", ""))]
    if not kd:
        print(f"[plot] no KD rows matching {pattern!r}; skipping {fname}.")
        return
    temps = sorted({_f(r, "temperature") for r in kd})
    alphas = sorted({_f(r, "alpha") for r in kd})
    grid = [[float("nan")] * len(alphas) for _ in temps]
    for r in kd:
        i = temps.index(_f(r, "temperature"))
        j = alphas.index(_f(r, "alpha"))
        grid[i][j] = _f(r, "best_top1")

    plt.figure(figsize=(5, 4))
    im = plt.imshow(grid, aspect="auto", origin="lower")
    plt.colorbar(im, label="top-1 (%)")
    for i in range(len(temps)):
        for j in range(len(alphas)):
            if grid[i][j] == grid[i][j]:  # not NaN
                plt.text(j, i, f"{grid[i][j]:.1f}", ha="center", va="center",
                         fontsize=8, color="w")
    plt.xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    plt.yticks(range(len(temps)), [f"{t:g}" for t in temps])
    plt.xlabel("alpha")
    plt.ylabel("temperature T")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out / fname, dpi=150)
    plt.close()
    print(f"[plot] wrote {out / fname}")


def plot_kd_gain(rows, out: Path) -> None:
    pairs, _ = final_pairs(rows)
    if not pairs:
        print("[plot] no final baseline/KD pairs; skipping kd_gain.")
        return
    sizes = [p[1] for p in pairs]
    gains = [p[3] - p[2] for p in pairs]
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, gains, marker="o")
    for (label, x, b, k), g in zip(pairs, gains):
        plt.annotate(label, (x, g), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("student params (M)")
    plt.ylabel("KD gain: top-1(KD) - top-1(baseline)  (pp)")
    plt.title("When does KD help the most?")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "kd_gain.png", dpi=150)
    plt.close()
    print(f"[plot] wrote {out / 'kd_gain.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 plots")
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    rows = read_results(args.csv)
    if not rows:
        print(f"[plot] no rows in {args.csv}; run some training first.")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_pareto(rows, out)
    # final setup: no label smoothing, adamw recipe, MBV2 w0.5, seed 42
    plot_heatmap(rows, out, r"kd_cifar100_w0\.5_adamw1e-3_T[\d.]+_a[\d.]+_nols",
                 "heatmap.png", "KD on MBV2 w0.5 (no LS, adamw)")
    # old Night-2 sweep (label-smoothed teacher, sgd 0.01) -> shows the high-alpha collapse
    plot_heatmap(rows, out, r"kd_cifar100_w0\.5_T[\d.]+_a[\d.]+",
                 "heatmap_old_ls.png", "Old sweep: LS teacher + sgd0.01 (w0.5)")
    plot_kd_gain(rows, out)


if __name__ == "__main__":
    main()
