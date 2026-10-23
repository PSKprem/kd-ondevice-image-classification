"""P25 — export a trained checkpoint: PyTorch -> ONNX -> TFLite (FP32 + INT8).

The ONNX->TF->TFLite toolchain is version-fragile on Windows (onnx-tf /
tensorflow). If it fails here, run this step in Colab/WSL as noted in task.md;
the ONNX file produced in step 1 is portable.

Usage:
    python scripts/export.py --ckpt D:/kd-project/runs/<run>/best.pt \
        --config configs/student.yaml --out D:/kd-project/export/<run>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import build_model  # noqa: E402
from src.utils import load_checkpoint, load_config  # noqa: E402

NUM_CLASSES = {"cifar100": 100}


def to_onnx(model, image_size: int, onnx_path: Path) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    print(f"[export] wrote ONNX -> {onnx_path}")


def onnx_to_tflite(onnx_path: Path, out_dir: Path, image_size: int) -> None:
    """ONNX -> TF SavedModel -> TFLite FP32 + INT8. Best-effort on Windows."""
    try:
        import numpy as np
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError as e:
        print(f"[export] TFLite toolchain unavailable ({e}). "
              "Do ONNX->TFLite in Colab/WSL (see task.md).")
        return

    saved_model_dir = out_dir / "saved_model"
    onnx_model = onnx.load(str(onnx_path))
    prepare(onnx_model).export_graph(str(saved_model_dir))
    print(f"[export] wrote TF SavedModel -> {saved_model_dir}")

    # FP32
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    fp32 = conv.convert()
    (out_dir / "model_fp32.tflite").write_bytes(fp32)
    print(f"[export] wrote {out_dir / 'model_fp32.tflite'}")

    # INT8 (dynamic-range quantization; simplest, no rep-dataset needed)
    conv_int8 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    int8 = conv_int8.convert()
    (out_dir / "model_int8.tflite").write_bytes(int8)
    print(f"[export] wrote {out_dir / 'model_int8.tflite'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 export")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--width-mult", type=float, default=None)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--onnx-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    width = args.width_mult if args.width_mult is not None else float(cfg.get("width_mult", 1.0))
    image_size = int(cfg["image_size"])
    num_classes = NUM_CLASSES.get(str(cfg["dataset"]).lower(), 100)

    model = build_model(cfg["model"], num_classes, width_mult=width, pretrained=False)
    load_checkpoint(args.ckpt, model, map_location="cpu")
    model.eval()

    out_dir = Path(args.out)
    onnx_path = out_dir / "model.onnx"
    to_onnx(model, image_size, onnx_path)

    if not args.onnx_only:
        onnx_to_tflite(onnx_path, out_dir, image_size)


if __name__ == "__main__":
    main()
