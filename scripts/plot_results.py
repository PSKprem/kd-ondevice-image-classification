"""P25 — plots + summary table from the v2 results (results/results_v2.csv).

Headline metric = test_top1 of the val-selected checkpoint (see README, "Evaluation protocol (v2)").
The v1 file (results/results.csv) chose the best epoch on test and is NOT used here.

Produces (into results/figures/):
  - pareto_v2.png      test top-1 vs params: baseline vs KD (+ teacher reference line)
  - kd_gain_v2.png     KD gain (pp) vs params (error bars = std over seeds when >1 seed exists)
  - heatmap_v2.png     T x alpha grid on w1.0: val (used for selection) and test (reported)
  - alpha_probe_v2.png KD gain vs alpha at T=2 per student (only students with >=2 alphas)
and results/summary_table_v2.csv (+ a markdown table printed to the terminal).

Usage (from repo root):
    python scripts/plot_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import read_results  # noqa: E402

MAIN_SEED = 42
# tag -> (label, model, width_mult)
STUDENTS = {
    "w0.5": ("MBV2 w0.5", "mobilenetv2", 0.5),
    "mbv3s": ("MBV3-Small", "mobilenetv3_small", 1.0),
    "w1.0": ("MBV2 w1.0", "mobilenetv2", 1.0),
    "w1.4": ("MBV2 w1.4", "mobilenetv2", 1.4),
}
RECIPE_OPT = {"sgd0.01": "sgd", "adamw1e-3": "adamw"}  # recipe name -> optimizer column value


def _f(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def find(rows, tag, recipe, kd, T=None, alpha=None, seed=MAIN_SEED):
    """First row matching student / recipe / kd flag (/ T / alpha) / seed, else None."""
    _, model, width = STUDENTS[tag]
    for r in rows:
        if (r.get("model") == model and abs(_f(r, "width_mult") - width) < 1e-9
                and int(_f(r, "kd", 0)) == kd and r.get("optimizer") == RECIPE_OPT[recipe]
                and int(_f(r, "seed", -1)) == seed and int(_f(r, "pretrained", 1)) == 1):
            if kd and (abs(_f(r, "temperature") - T) > 1e-9 or abs(_f(r, "alpha") - alpha) > 1e-9):
                continue
            return r
    return None


def teacher_row(rows):
    for r in rows:
        if r.get("model") == "resnet50" and int(_f(r, "kd", 0)) == 0 and int(_f(r, "seed", -1)) == MAIN_SEED:
            return r
    return None


def paired(rows, tag, recipe, T, a):
    """[(baseline_test, kd_test)] over seeds where both runs exist."""
    out = []
    for seed in (MAIN_SEED, 0, 1):
        b, k = find(rows, tag, recipe, 0, seed=seed), find(rows, tag, recipe, 1, T, a, seed=seed)
        if b and k:
            out.append((_f(b, "test_top1"), _f(k, "test_top1")))
    return out


def summary(rows, recipes, T, a):
    """One dict per student that has a finished baseline + KD pair."""
    res = []
    for tag, (label, _, _) in STUDENTS.items():
        if tag not in recipes:
            continue
        pairs = paired(rows, tag, recipes[tag], T, a)
        if not pairs:
            continue
        gains = [k - b for b, k in pairs]
        b0 = find(rows, tag, recipes[tag], 0)
        res.append({
            "tag": tag, "label": label, "recipe": recipes[tag], "params_m": _f(b0, "params_m"),
            "n_seeds": len(pairs),
            "base": statistics.mean(b for b, _ in pairs), "kd": statistics.mean(k for _, k in pairs),
            "gain": statistics.mean(gains),
            "gain_std": statistics.stdev(gains) if len(gains) > 1 else 0.0,
        })
    return sorted(res, key=lambda d: d["params_m"])


def plot_pareto(summ, teacher, T, a, out: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    xs = [s["params_m"] for s in summ]
    plt.plot(xs, [s["base"] for s in summ], "o--", label="baseline (no KD)")
    plt.plot(xs, [s["kd"] for s in summ], "^-", label=f"KD (T={T:g}, α={a:g})")
    for s in summ:
        plt.annotate(f"{s['label']}\n{s['gain']:+.2f}", (s["params_m"], s["kd"]),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    if teacher:  # teacher drawn as a reference line (23.7M params would squash the x-axis)
        ty = _f(teacher, "test_top1")
        plt.axhline(ty, color="gray", ls=":", lw=1)
        plt.text(0.01, ty + 0.15, f"teacher ResNet-50 ({_f(teacher, 'params_m'):.1f}M): {ty:.2f}",
                 transform=plt.gca().get_yaxis_transform(), ha="left", fontsize=8, color="gray")
    plt.xlabel("student params (M)")
    plt.ylabel("CIFAR-100 test top-1 (%)")
    plt.title("Accuracy vs model size: baseline vs KD")
    plt.margins(x=0.1)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "pareto_v2.png", dpi=150)
    plt.close()
    print(f"[plot] wrote {out / 'pareto_v2.png'}")


def plot_kd_gain(summ, out: Path) -> None:
    plt.figure(figsize=(6, 4))
    xs, ys = [s["params_m"] for s in summ], [s["gain"] for s in summ]
    errs = [s["gain_std"] for s in summ]
    if any(e > 0 for e in errs):
        plt.errorbar(xs, ys, yerr=errs, marker="o", capsize=3)
    else:
        plt.plot(xs, ys, marker="o")
    for s in summ:
        plt.annotate(f"{s['label']} (n={s['n_seeds']})", (s["params_m"], s["gain"]),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("student params (M)")
    plt.ylabel("KD gain: test top-1(KD) − top-1(baseline)  (pp)")
    plt.title("When does KD help the most?")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "kd_gain_v2.png", dpi=150)
    plt.close()
    print(f"[plot] wrote {out / 'kd_gain_v2.png'}")


def plot_heatmap(rows, tag, recipe, out: Path) -> None:
    kd = [r for r in rows if int(_f(r, "kd", 0)) == 1 and int(_f(r, "seed", -1)) == MAIN_SEED
          and find([r], tag, recipe, 1, _f(r, "temperature"), _f(r, "alpha"))]
    if not kd:
        print(f"[plot] no KD grid for {tag}; skipping heatmap.")
        return
    temps = sorted({_f(r, "temperature") for r in kd})
    alphas = sorted({_f(r, "alpha") for r in kd})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(axes, ("best_val_top1", "test_top1"),
                              ("val top-1 (used for selection)", "test top-1 (reported)")):
        grid = [[float("nan")] * len(alphas) for _ in temps]
        for r in kd:
            grid[temps.index(_f(r, "temperature"))][alphas.index(_f(r, "alpha"))] = _f(r, key)
        im = ax.imshow(grid, aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax)
        for i in range(len(temps)):
            for j in range(len(alphas)):
                if grid[i][j] == grid[i][j]:
                    ax.text(j, i, f"{grid[i][j]:.2f}", ha="center", va="center", fontsize=8, color="w")
        ax.set_xticks(range(len(alphas)), [f"{x:g}" for x in alphas])
        ax.set_yticks(range(len(temps)), [f"{t:g}" for t in temps])
        ax.set_xlabel("alpha")
        ax.set_ylabel("temperature T")
        ax.set_title(title, fontsize=10)
    b = find(rows, tag, recipe, 0)
    fig.suptitle(f"KD on {STUDENTS[tag][0]} ({recipe}); baseline val {_f(b, 'best_val_top1'):.2f} / "
                 f"test {_f(b, 'test_top1'):.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "heatmap_v2.png", dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out / 'heatmap_v2.png'}")


def plot_alpha_probe(rows, recipes, T, out: Path) -> None:
    plt.figure(figsize=(6, 4))
    drawn = 0
    for tag, (label, _, _) in STUDENTS.items():
        if tag not in recipes:
            continue
        b = find(rows, tag, recipes[tag], 0)
        pts = []
        for r in rows:
            if find([r], tag, recipes[tag], 1, T, _f(r, "alpha")) and int(_f(r, "seed", -1)) == MAIN_SEED:
                pts.append((_f(r, "alpha"), _f(r, "test_top1") - _f(b, "test_top1")))
        pts = sorted(set(pts))
        if len(pts) >= 2:
            plt.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=label)
            drawn += 1
    if not drawn:
        plt.close()
        print("[plot] no student with >=2 alphas at the common T; skipping alpha probe.")
        return
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("alpha")
    plt.ylabel("KD gain on test (pp)")
    plt.title(f"alpha sensitivity at T={T:g} (single seed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "alpha_probe_v2.png", dpi=150)
    plt.close()
    print(f"[plot] wrote {out / 'alpha_probe_v2.png'}")


def write_table(summ, teacher, T, a) -> None:
    path = ROOT / "results" / "summary_table_v2.csv"
    fields = ["model", "params_m", "recipe", "n_seeds", "baseline_test", "kd_test", "kd_gain_pp", "gain_std_pp"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        if teacher:
            w.writerow(["ResNet-50 (teacher)", f"{_f(teacher, 'params_m'):.2f}", "sgd0.01", 1,
                        f"{_f(teacher, 'test_top1'):.2f}", "", "", ""])
        for s in summ:
            w.writerow([s["label"], f"{s['params_m']:.2f}", s["recipe"], s["n_seeds"], f"{s['base']:.2f}",
                        f"{s['kd']:.2f}", f"{s['gain']:+.2f}", f"{s['gain_std']:.2f}"])
    print(f"[plot] wrote {path}")
    print(f"\n| Model | Params (M) | n seeds | Baseline test | KD test (T={T:g}, a={a:g}) | Gain (pp) |")
    print("|---|---|---|---|---|---|")
    if teacher:
        print(f"| ResNet-50 (teacher) | {_f(teacher, 'params_m'):.2f} | 1 | {_f(teacher, 'test_top1'):.2f} | - | - |")
    for s in summ:
        print(f"| {s['label']} | {s['params_m']:.2f} | {s['n_seeds']} | {s['base']:.2f} | {s['kd']:.2f} | {s['gain']:+.2f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 v2 plots")
    ap.add_argument("--csv", default="results/results_v2.csv")
    ap.add_argument("--config", default="results/best_config_v2.json", help="source of per-student recipes")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--T", type=float, default=2.0, help="common KD temperature")
    ap.add_argument("--alpha", type=float, default=0.9, help="common KD alpha")
    args = ap.parse_args()

    rows = read_results(args.csv)
    if not rows:
        print(f"[plot] no rows in {args.csv}; run some training first.")
        return
    recipes = json.loads(Path(args.config).read_text(encoding="utf-8"))["recipes"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summ = summary(rows, recipes, args.T, args.alpha)
    teacher = teacher_row(rows)
    if summ:
        plot_pareto(summ, teacher, args.T, args.alpha, out)
        plot_kd_gain(summ, out)
        write_table(summ, teacher, args.T, args.alpha)
    else:
        print("[plot] no finished baseline/KD pairs yet; skipping pareto / kd_gain / table.")
    plot_heatmap(rows, "w1.0", recipes["w1.0"], out)
    plot_alpha_probe(rows, recipes, args.T, out)


if __name__ == "__main__":
    main()
