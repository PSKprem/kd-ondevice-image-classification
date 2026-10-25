"""P25 — accuracy of an exported .tflite model on the test set (+ agreement with the PyTorch checkpoint).

Run in the EXPORT venv (D:\\kd-project\\.venv-export). Uses the same preprocessing as training
(src.data eval transform), so the number is directly comparable to `test_top1` in results_v2.csv.

Usage:
    D:\\kd-project\\.venv-export\\Scripts\\python.exe scripts/eval_tflite.py ^
        --tflite D:/kd-project/export/test_w1.0/tf/model_float32.tflite ^
        --config configs/student.yaml --ckpt D:/kd-project/runs_v2/<run>/best.pt --width-mult 1.0 --n 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_datasets  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_checkpoint, load_config  # noqa: E402


def load_interpreter(path: str):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # fall back to TensorFlow's copy
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    return it


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 TFLite accuracy check")
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="PyTorch checkpoint to compare predictions with")
    ap.add_argument("--width-mult", type=float, default=None)
    ap.add_argument("--n", type=int, default=2000, help="number of test images (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _, _, test_set, num_classes = build_datasets(cfg)
    n = len(test_set) if args.n <= 0 else min(args.n, len(test_set))

    it = load_interpreter(args.tflite)
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    shape = list(inp["shape"])
    nhwc = shape[-1] == 3  # onnx2tf converts NCHW -> NHWC
    print(f"[eval] tflite input {shape} ({'NHWC' if nhwc else 'NCHW'}) dtype={inp['dtype'].__name__}")

    torch_model = None
    if args.ckpt:
        width = args.width_mult if args.width_mult is not None else float(cfg.get("width_mult", 1.0))
        torch_model = build_model(cfg["model"], num_classes, width_mult=width, pretrained=False)
        load_checkpoint(args.ckpt, torch_model, map_location="cpu")
        torch_model.eval()

    correct = agree = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            x, y = test_set[i]
            xb = x.unsqueeze(0).numpy()
            if nhwc:
                xb = np.transpose(xb, (0, 2, 3, 1))
            xb = np.ascontiguousarray(xb, dtype=np.float32)
            if np.issubdtype(inp["dtype"], np.integer):  # fully-quantized model: quantize the input
                scale, zero = inp["quantization"]
                info = np.iinfo(inp["dtype"])
                xb = np.clip(np.round(xb / scale + zero), info.min, info.max)
            it.set_tensor(inp["index"], xb.astype(inp["dtype"]))
            it.invoke()
            logits = it.get_tensor(out["index"]).astype(np.float32)  # argmax is scale-invariant for per-tensor out
            pred = int(np.argmax(logits))
            correct += int(pred == y)
            if torch_model is not None:
                agree += int(pred == int(torch_model(x.unsqueeze(0)).argmax(1)))
    msg = f"[eval] {Path(args.tflite).name}: top-1 = {100 * correct / n:.2f}% on {n} test images"
    if torch_model is not None:
        msg += f" | agreement with PyTorch = {100 * agree / n:.2f}%"
    print(msg + f" ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
