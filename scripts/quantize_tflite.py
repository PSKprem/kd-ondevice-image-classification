"""P25 — INT8-quantize a float32 .tflite with Google's ai-edge-quantizer (export venv).

Why this route: onnx2tf's own quantized outputs were broken for MobileNet on Windows (dynamic-range
model gave ~1.5% accuracy; tf_converter backend failed on a layout bug). Quantizing the already
verified FP32 .tflite flatbuffer avoids the ONNX->TF step entirely.

Modes
  dynamic    : weights int8 (channel-wise), activations quantized at run time
  weightonly : weights int8 (smaller file), computation in float32 -> accuracy ~ FP32, no int8 speed-up
  static     : weights int8 + activations int8 (needs --calib, an .npy of NHWC float32 images)

Usage (export venv):
    D:\\kd-project\\.venv-export\\Scripts\\python.exe scripts/quantize_tflite.py ^
        --fp32 D:/kd-project/export/<run>/tf/model_float32.tflite --out D:/kd-project/export/<run>/model_int8_dynamic.tflite
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ai_edge_quantizer import quantizer, recipe


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 TFLite INT8 quantization")
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["dynamic", "static", "weightonly", "mixed_weightonly", "mixed_dynamic"],
                    default="dynamic")
    ap.add_argument("--calib", default=None, help=".npy NHWC float32 images (static mode)")
    args = ap.parse_args()

    qt = quantizer.Quantizer(float_model=args.fp32)
    if args.mode == "dynamic":
        qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
        result = qt.quantize()
    elif args.mode.startswith("mixed"):
        # depthwise convs stay float32 (they are the layers that lose accuracy when quantized, see scripts/diag_quant.py);
        # regular convs + fully-connected layers are int8 (weight-only or dynamic).
        base = (recipe.weight_only_wi8_afp32() if args.mode == "mixed_weightonly" else recipe.dynamic_wi8_afp32())[0]
        rules = []
        for op in ("CONV_2D", "FULLY_CONNECTED"):
            rule = dict(base)
            rule["operation"] = op
            rules.append(rule)
        qt.load_quantization_recipe(rules)
        result = qt.quantize()
    elif args.mode == "weightonly":
        qt.load_quantization_recipe(recipe.weight_only_wi8_afp32())
        result = qt.quantize()
    else:
        if not args.calib:
            raise SystemExit("--calib is required for static mode")
        qt.load_quantization_recipe(recipe.static_wi8_ai8())
        calib = np.load(args.calib).astype(np.float32)
        from ai_edge_litert.interpreter import Interpreter
        sigs = Interpreter(model_path=args.fp32).get_signature_list()
        sig_key = next(iter(sigs))  # onnx2tf writes a single "serving_default" signature
        input_name = sigs[sig_key]["inputs"][0]
        data = {sig_key: [{input_name: calib[i:i + 1]} for i in range(len(calib))]}
        cal_result = qt.calibrate(data)
        result = qt.quantize(cal_result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.export_model(str(out), overwrite=True)
    print(f"[quant] {args.mode}: {Path(args.fp32).stat().st_size / 1e6:.2f} MB -> {out.stat().st_size / 1e6:.2f} MB  ({out})")


if __name__ == "__main__":
    main()
