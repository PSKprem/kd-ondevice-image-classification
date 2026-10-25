"""P25 — extra figures for the report (from the committed CSVs; no training needed).

  fig_latency_pareto.png  test accuracy vs on-device latency (Galaxy F22, 1 thread): FP32 vs INT8-mixed, KD students + teacher
  fig_lowdata.png         KD gain vs fraction of the training data (w0.5, w1.0)
  fig_scratch.png         KD gain: pretrained vs from scratch (w0.5, w1.0)
  fig_optimizer.png       KD gain under SGD and AdamW for every student

Usage (from repo root):  python scripts/plot_report_figures.py
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"


def rd(name: str) -> list[dict]:
    with open(ROOT / "results" / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_latency() -> None:
    lat = defaultdict(list)
    for r in rd("phone_latency_v2.csv"):
        if r["threads"] == "1":
            lat[(r["student"], r["precision"])].append(float(r["avg_ms"]))
    med = {k: st.median(v) for k, v in lat.items()}
    # test accuracy: FP32 = PyTorch KD run (seed 42, results_v2.csv); INT8 mixed = TFLite on the full 10k test set
    fp32 = {"w0.5": 75.03, "mbv3s": 80.02, "w1.0": 81.71, "w1.4": 83.61, "teacher": 85.46}
    int8 = {}
    for r in rd("export_eval_v2.csv"):
        if r["tflite"] == "model_int8_mixed_dynamic.tflite" and r["tag"].startswith("kd_") and r["n_images"] == "10000":
            tag = {"w0.5": "w0.5", "mbv3s": "mbv3s", "w1.0": "w1.0", "w1.4": "w1.4"}
            for k in tag:
                if f"_{k}_" in r["tag"]:
                    int8[k] = float(r["tflite_top1"])
    label = {"w0.5": "MBV2 w0.5", "mbv3s": "MBV3-Small", "w1.0": "MBV2 w1.0", "w1.4": "MBV2 w1.4", "teacher": "ResNet-50 teacher"}
    plt.figure(figsize=(7.5, 4.6))
    plt.plot([med[(k, "FP32")] for k in fp32], [fp32[k] for k in fp32], "o", color="tab:blue", label="FP32")
    plt.plot([med[(k, "INT8 mixed")] for k in int8], [int8[k] for k in int8], "^", color="tab:orange", label="INT8 mixed (depthwise float)")
    for k in fp32:
        plt.annotate(label[k], (med[(k, "FP32")], fp32[k]), textcoords="offset points", xytext=(6, -12), fontsize=8)
    for k in int8:
        plt.plot([med[(k, "FP32")], med[(k, "INT8 mixed")]], [fp32[k], int8[k]], "-", color="gray", lw=0.7)
    plt.xscale("log")
    plt.xlabel("latency on phone, 1 CPU thread (ms, log scale; median of 3 repetitions)")
    plt.ylabel("CIFAR-100 test top-1 (%)")
    plt.title("Accuracy vs on-device latency (Samsung Galaxy F22, KD students)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUT / "fig_latency_pareto.png", dpi=150)
    plt.close()


def fig_size_pareto() -> None:
    """Accuracy vs model FILE size (MiB): baseline vs KD, FP32 vs INT8-mixed (seed-42 models)."""
    ex = rd("export_eval_v2.csv")
    def student_of(tag: str) -> str:
        return "mbv3s" if "mbv3s" in tag else next(k for k in ("w0.5", "w1.0", "w1.4") if f"_{k}_" in tag)
    mixed, fp_size = {}, {}
    for r in ex:
        kind = "kd" if r["tag"].startswith("kd_") else "baseline" if r["tag"].startswith("student_") else "teacher"
        if kind == "teacher":
            continue
        s = student_of(r["tag"])
        if r["tflite"] == "model_int8_mixed_dynamic.tflite" and r["n_images"] == "10000":
            mixed[(kind, s)] = (float(r["size_mib"]), float(r["tflite_top1"]))
        if r["tflite"] == "model_float32.tflite" and kind == "kd":
            fp_size[s] = float(r["size_mib"])
    fp_acc = {("baseline", "w0.5"): 74.58, ("baseline", "mbv3s"): 78.34, ("baseline", "w1.0"): 80.86, ("baseline", "w1.4"): 82.36,
              ("kd", "w0.5"): 75.03, ("kd", "mbv3s"): 80.02, ("kd", "w1.0"): 81.71, ("kd", "w1.4"): 83.61}
    order = ["w0.5", "mbv3s", "w1.0", "w1.4"]
    plt.figure(figsize=(7.5, 4.6))
    for kind, mk, col in (("baseline", "o", "tab:gray"), ("kd", "^", "tab:green")):
        name = "baseline" if kind == "baseline" else "distilled (KD)"
        plt.plot([fp_size[s] for s in order], [fp_acc[(kind, s)] for s in order], mk + "--", color=col, alpha=0.6, label=f"{name}, FP32")
        plt.plot([mixed[(kind, s)][0] for s in order], [mixed[(kind, s)][1] for s in order], mk + "-", color=col, label=f"{name}, INT8 mixed")
    plt.plot([90.39], [85.46], "s", color="tab:red", label="ResNet-50 teacher, FP32")
    plt.annotate("teacher", (90.39, 85.46), textcoords="offset points", xytext=(-38, -4), fontsize=8)
    labels = {"w0.5": "w0.5", "mbv3s": "MBV3-S", "w1.0": "w1.0", "w1.4": "w1.4"}
    for s in order:
        plt.annotate(labels[s], (mixed[("kd", s)][0], mixed[("kd", s)][1]), textcoords="offset points", xytext=(4, 6), fontsize=8)
    plt.xscale("log")
    plt.xlabel("model file size (MiB, log scale)")
    plt.ylabel("CIFAR-100 test top-1 (%)")
    plt.title("Accuracy vs model size (file on disk): baseline vs KD, FP32 vs INT8")
    plt.legend(fontsize=7, loc="lower right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUT / "fig_size_pareto.png", dpi=150)
    plt.close()


def fig_lowdata() -> None:
    rows = rd("summary_lowdata_v2.csv")
    plt.figure(figsize=(6, 4))
    for tag, name in (("w1.0", "MBV2 w1.0 (SGD)"), ("w0.5", "MBV2 w0.5 (AdamW)")):
        pts = sorted((float(r["train_fraction"]), float(r["kd_gain_pp"])) for r in rows if r["student"] == tag)
        plt.plot([p[0] * 100 for p in pts], [p[1] for p in pts], "o-", label=name)
    plt.axhline(0, color="k", lw=0.8)
    plt.xscale("log")
    plt.xticks([10, 25, 100], ["10%", "25%", "100%"])
    plt.xlabel("fraction of the CIFAR-100 training data used")
    plt.ylabel("KD gain on test (pp)")
    plt.title("KD gain vs amount of training data (single seed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig_lowdata.png", dpi=150)
    plt.close()


def fig_scratch() -> None:
    sc = {r["student"]: float(r["kd_gain_pp"]) for r in rd("summary_scratch_v2.csv")}
    kd = {r["student"]: float(r["kd_gain_pp"]) for r in rd("kd_gain_v2.csv")}
    tags = ["w0.5", "w1.0"]
    x = range(len(tags))
    plt.figure(figsize=(5.5, 4))
    plt.bar([i - 0.2 for i in x], [kd[t] for t in tags], 0.4, label="pretrained (3 seeds)")
    plt.bar([i + 0.2 for i in x], [sc[t] for t in tags], 0.4, label="from scratch (1 seed)")
    for i, t in enumerate(tags):
        plt.text(i - 0.2, kd[t] + 0.1, f"+{kd[t]:.2f}", ha="center", fontsize=8)
        plt.text(i + 0.2, sc[t] + 0.1, f"+{sc[t]:.2f}", ha="center", fontsize=8)
    plt.xticks(list(x), ["MBV2 w0.5", "MBV2 w1.0"])
    plt.ylabel("KD gain on test (pp)")
    plt.title("KD helps far more without ImageNet weights")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "fig_scratch.png", dpi=150)
    plt.close()


def fig_optimizer() -> None:
    rows = rd("recipe_swap_v2.csv")
    order = [("w0.5", "MBV2 w0.5"), ("mbv3s", "MBV3-Small"), ("w1.0", "MBV2 w1.0"), ("w1.4", "MBV2 w1.4")]
    g = {(r["student"], r["recipe"]): float(r["test_gain_pp"]) for r in rows}
    x = range(len(order))
    plt.figure(figsize=(6.5, 4))
    plt.bar([i - 0.2 for i in x], [g[(t, "sgd0.01")] for t, _ in order], 0.4, label="SGD 0.01")
    plt.bar([i + 0.2 for i in x], [g[(t, "adamw1e-3")] for t, _ in order], 0.4, label="AdamW 1e-3")
    for i, (t, _) in enumerate(order):
        plt.text(i - 0.2, g[(t, "sgd0.01")] + 0.08, f"{g[(t, 'sgd0.01')]:+.2f}", ha="center", fontsize=8)
        plt.text(i + 0.2, g[(t, "adamw1e-3")] + 0.08, f"{g[(t, 'adamw1e-3')]:+.2f}", ha="center", fontsize=8)
    plt.xticks(list(x), [n for _, n in order], fontsize=8)
    plt.ylabel("KD gain on test (pp, seed 42)")
    plt.title("KD gain under both optimizers (SGD baseline of w0.5 is under-trained)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "fig_optimizer.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_latency, fig_size_pareto, fig_lowdata, fig_scratch, fig_optimizer):
        fn()
        print("[fig]", fn.__name__)
