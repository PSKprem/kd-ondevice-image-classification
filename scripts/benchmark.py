"""P25 — model cost benchmark: params, FLOPs, checkpoint file size.

Optionally measures on-device latency of a .tflite model via the Android
`benchmark_model` binary pushed through adb (only if --tflite and adb are set up).

Usage:
    python scripts/benchmark.py --config configs/student.yaml --width-mult 0.5
    python scripts/benchmark.py --ckpt D:/kd-project/runs/<run>/best.pt --config configs/student.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import build_model  # noqa: E402
from src.utils import count_params_millions, load_config  # noqa: E402

NUM_CLASSES = {"cifar100": 100}


def measure_flops(model, image_size: int) -> float | None:
    """GFLOPs for one forward pass; needs `thop`. Returns None if unavailable."""
    try:
        from thop import profile
    except ImportError:
        print("[bench] thop not installed; skipping FLOPs.")
        return None
    dummy = torch.randn(1, 3, image_size, image_size)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return macs * 2 / 1e9  # MACs -> FLOPs -> giga


def file_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def adb_latency(tflite_path: str, runs: int = 100) -> None:
    """Push a tflite model to the phone and run benchmark_model via adb."""
    device_path = "/data/local/tmp/model.tflite"
    try:
        subprocess.run(["adb", "push", tflite_path, device_path], check=True)
        out = subprocess.run(
            ["adb", "shell", "/data/local/tmp/benchmark_model",
             f"--graph={device_path}", f"--num_runs={runs}"],
            check=True, capture_output=True, text=True,
        )
        print(out.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[bench] adb latency skipped/failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 benchmark")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint for file-size report")
    ap.add_argument("--width-mult", type=float, default=None)
    ap.add_argument("--tflite", default=None, help="tflite model for adb latency")
    args = ap.parse_args()

    cfg = load_config(args.config)
    width = args.width_mult if args.width_mult is not None else float(cfg.get("width_mult", 1.0))
    image_size = int(cfg["image_size"])
    num_classes = NUM_CLASSES.get(str(cfg["dataset"]).lower(), 100)

    model = build_model(cfg["model"], num_classes, width_mult=width, pretrained=False)
    model.eval()

    params_m = count_params_millions(model)
    gflops = measure_flops(model, image_size)
    print(f"[bench] model={cfg['model']} width={width} image_size={image_size}")
    print(f"[bench] params = {params_m:.3f} M")
    print(f"[bench] FLOPs  = {gflops:.3f} GFLOPs" if gflops is not None else "[bench] FLOPs  = n/a")

    if args.ckpt and Path(args.ckpt).exists():
        print(f"[bench] ckpt size = {file_size_mb(args.ckpt):.2f} MB")

    if args.tflite:
        if Path(args.tflite).exists():
            print(f"[bench] tflite size = {file_size_mb(args.tflite):.2f} MB")
        adb_latency(args.tflite)


if __name__ == "__main__":
    main()
