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


def plot_pareto(rows, out: Path) -> None:
    base = [r for r in rows if int(_f(r, "kd")) == 0 and r.get("model") != "resnet50"]
    kd = [r for r in rows if int(_f(r, "kd")) == 1]
    plt.figure(figsize=(6, 4))
    if base:
        plt.scatter([_f(r, "params_m") for r in base],
                    [_f(r, "best_top1") for r in base],
                    marker="o", label="baseline (no KD)")
    if kd:
        plt.scatter([_f(r, "params_m") for r in kd],
                    [_f(r, "best_top1") for r in kd],
                    marker="^", label="KD")
    plt.xlabel("params (M)")
    plt.ylabel("top-1 accuracy (%)")
    plt.title("Accuracy vs model size (Pareto)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "pareto.png", dpi=150)
    plt.close()
    print(f"[plot] wrote {out / 'pareto.png'}")


def plot_heatmap(rows, out: Path) -> None:
    kd = [r for r in rows if int(_f(r, "kd")) == 1]
    if not kd:
        print("[plot] no KD rows; skipping heatmap.")
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
    plt.xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    plt.yticks(range(len(temps)), [f"{t:g}" for t in temps])
    plt.xlabel("alpha")
    plt.ylabel("temperature T")
    plt.title("KD sweep: top-1 over T x alpha")
    plt.tight_layout()
    plt.savefig(out / "heatmap.png", dpi=150)
    plt.close()
    print(f"[plot] wrote {out / 'heatmap.png'}")


def plot_kd_gain(rows, out: Path) -> None:
    base = {}
    for r in rows:
        if int(_f(r, "kd")) == 0 and r.get("model") != "resnet50":
            base[_f(r, "width_mult")] = _f(r, "best_top1")
    widths, gains = [], []
    for r in rows:
        if int(_f(r, "kd")) == 1:
            w = _f(r, "width_mult")
            if w in base:
                widths.append(w)
                gains.append(_f(r, "best_top1") - base[w])
    if not widths:
        print("[plot] no matched baseline/KD widths; skipping kd_gain.")
        return
    order = sorted(range(len(widths)), key=lambda i: widths[i])
    widths = [widths[i] for i in order]
    gains = [gains[i] for i in order]
    plt.figure(figsize=(6, 4))
    plt.plot(widths, gains, marker="o")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("MobileNetV2 width")
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
    plot_heatmap(rows, out)
    plot_kd_gain(rows, out)


if __name__ == "__main__":
    main()
