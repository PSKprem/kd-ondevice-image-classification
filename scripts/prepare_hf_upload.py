"""P25 — build the Hugging Face upload folder (final models only) and its model card from the committed CSVs.

Output: D:/kd-project/hf_upload/
  README.md                      model card (numbers read from results/*.csv)
  classes.json                   CIFAR-100 class names in label order (torchvision order)
  models/<name>/model.safetensors   weights only (state_dict of best.pt; optimizer state dropped)
  models/<name>/config.json         architecture + training info
  models/<name>/model_fp32.tflite   exported FP32 model (input NHWC float32 [1,128,128,3])
  models/<name>/model_int8_mixed.tflite   INT8 with depthwise convs kept float32 (students only)

Each safetensors file is re-loaded into a fresh model and compared tensor-by-tensor with best.pt.

Usage (training venv, from repo root):  python scripts/prepare_hf_upload.py
"""
from __future__ import annotations

import csv
import json
import pickle
import shutil
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_model  # noqa: E402

RUNS = Path("D:/kd-project/runs_v2")
EXPORT = Path("D:/kd-project/export")
OUT = Path("D:/kd-project/hf_upload")
GITHUB = "https://github.com/PSKprem/kd-ondevice-image-classification"

# folder name -> (run name, model, width, phone/student tag, label)
MODELS = {
    "mbv2_w0.5_kd": ("kd_cifar100_w0.5_v2_adamw1e-3_T2_a0.9", "mobilenetv2", 0.5, "w0.5", "MobileNetV2 w0.5, distilled"),
    "mbv3_small_kd": ("kd_cifar100_mbv3s_v2_adamw1e-3_T2_a0.9", "mobilenetv3_small", 1.0, "mbv3s", "MobileNetV3-Small, distilled"),
    "mbv2_w1.0_kd": ("kd_cifar100_w1.0_v2_sgd0.01_T2_a0.9", "mobilenetv2", 1.0, "w1.0", "MobileNetV2 w1.0, distilled"),
    "mbv2_w1.4_kd": ("kd_cifar100_w1.4_v2_sgd0.01_T2_a0.9", "mobilenetv2", 1.4, "w1.4", "MobileNetV2 w1.4, distilled"),
    "resnet50_teacher": ("teacher_resnet50_cifar100_v2", "resnet50", 1.0, "teacher", "ResNet-50 teacher"),
}


def rd(name: str) -> list[dict]:
    with open(ROOT / "results" / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mib(p: Path) -> float:
    return p.stat().st_size / 2**20


def export_weights(folder: str, run: str, model_name: str, width: float, out: Path) -> dict:
    ckpt = torch.load(RUNS / run / "best.pt", map_location="cpu", weights_only=False)
    sd = {k: v.contiguous() for k, v in ckpt["model"].items()}
    save_file(sd, str(out / "model.safetensors"))
    # verify: fresh model + safetensors == best.pt weights
    m = build_model(model_name, 100, width_mult=width, pretrained=False)
    m.load_state_dict(load_file(str(out / "model.safetensors")))
    ref = ckpt["model"]
    same = all(torch.equal(v, ref[k]) for k, v in m.state_dict().items())
    if not same:
        raise RuntimeError(f"{folder}: safetensors weights differ from best.pt")
    return {"epoch": int(ckpt.get("epoch", -1)) + 1, "cfg": ckpt.get("config", {})}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)

    res = {r["run_name"]: r for r in rd("results_v2.csv")}
    ex = defaultdict(dict)
    for r in rd("export_eval_v2.csv"):
        ex[r["tag"]][(r["tflite"], r["n_images"])] = r
    lat = defaultdict(list)
    for r in rd("phone_latency_v2.csv"):
        lat[(r["student"], r["precision"], r["threads"])].append(float(r["avg_ms"]))
    seeds = {r["student"]: r for r in rd("kd_gain_v2.csv")}

    rows = []
    for folder, (run, model_name, width, tag, label) in MODELS.items():
        out = OUT / "models" / folder
        out.mkdir()
        info = export_weights(folder, run, model_name, width, out)
        shutil.copy2(EXPORT / run / "tf" / "model_float32.tflite", out / "model_fp32.tflite")
        has_int8 = tag != "teacher"
        if has_int8:
            shutil.copy2(EXPORT / run / "model_int8_mixed_dynamic.tflite", out / "model_int8_mixed.tflite")
        r = res[run]
        int8 = ex[run].get(("model_int8_mixed_dynamic.tflite", "10000")) if has_int8 else None
        cfg = info["cfg"]
        config = {
            "label": label, "architecture": model_name, "width_mult": width, "num_classes": 100,
            "dataset": "CIFAR-100 (45k train / 5k val split, split_seed 42; test = official 10k test set)",
            "input": {"size": [128, 128], "preprocess": "resize shorter side to 147, center crop 128, scale to [0,1], "
                      "normalize with ImageNet mean [0.485,0.456,0.406] / std [0.229,0.224,0.225]",
                      "pytorch_layout": "NCHW float32", "tflite_layout": "NHWC float32 [1,128,128,3]"},
            "training": {"distilled": tag != "teacher", "teacher": "resnet50_teacher" if tag != "teacher" else None,
                         "kd_temperature": float(r["temperature"]) if r["temperature"] else None,
                         "kd_alpha": float(r["alpha"]) if r["alpha"] else None,
                         "optimizer": r["optimizer"], "lr": float(r["lr"]), "weight_decay": float(r["weight_decay"]),
                         "epochs": int(r["epochs"]), "best_epoch": int(r["best_epoch"]), "seed": int(r["seed"]),
                         "imagenet_pretrained_init": bool(int(r["pretrained"])), "batch_size": cfg.get("batch_size")},
            "results_seed42": {"val_top1": float(r["best_val_top1"]), "test_top1_pytorch": float(r["test_top1"]),
                               "test_top1_tflite_int8_mixed": float(int8["tflite_top1"]) if int8 else None},
            "params_millions": round(float(r["params_m"]), 3),
            "source_run": run, "code": GITHUB,
        }
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        s = seeds.get(tag)
        rows.append({
            "folder": folder, "label": label, "params": float(r["params_m"]), "test": float(r["test_top1"]),
            "test3": f"{float(s['kd_test_mean']):.2f} ± {float(s['kd_test_std']):.2f}" if s else "—",
            "base3": f"{float(s['baseline_test_mean']):.2f} ± {float(s['baseline_test_std']):.2f}" if s else "—",
            "int8": float(int8["tflite_top1"]) if int8 else None,
            "fp32_mib": mib(out / "model_fp32.tflite"),
            "int8_mib": mib(out / "model_int8_mixed.tflite") if has_int8 else None,
            "lat_fp32": st.median(lat[(tag, "FP32", "1")]),
            "lat_int8": st.median(lat[(tag, "INT8 mixed", "1")]) if has_int8 else None,
        })
        print(f"[hf] {folder:18s} weights verified, files copied")

    # CIFAR-100 class names in label order (torchvision reads them from the dataset's meta file)
    meta = Path("D:/kd-project/data/cifar-100-python/meta")
    with open(meta, "rb") as f:
        classes = pickle.load(f, encoding="latin1")["fine_label_names"]
    (OUT / "classes.json").write_text(json.dumps(classes, indent=1), encoding="utf-8")

    def f(x, fmt="{:.2f}"):
        return "—" if x is None else fmt.format(x)

    table = "\n".join(
        f"| `{r['folder']}` | {r['label']} | {r['params']:.2f} | {r['base3']} | {r['test3']} | {r['test']:.2f} | {f(r['int8'])} "
        f"| {r['fp32_mib']:.2f} / {f(r['int8_mib'])} | {r['lat_fp32']:.2f} / {f(r['lat_int8'])} |" for r in rows)

    card = f"""---
tags:
- knowledge-distillation
- image-classification
- mobilenetv2
- mobilenetv3
- resnet50
- tflite
- on-device
datasets:
- cifar100
pipeline_tag: image-classification
---

# P25 — Knowledge Distillation for On-Device Image Classification (CIFAR-100)

Final models of course project **P25** (Section B, Team 25): a ResNet-50 teacher distilled (Hinton et al., 2015;
T = 2, α = 0.9) into MobileNet students, exported to TensorFlow Lite and measured on an Android phone.
Code, full results and the report: {GITHUB}

## Models

| Folder | Model | Params (M) | Baseline test top-1, 3 seeds | KD test top-1, 3 seeds | This checkpoint (seed 42), PyTorch | This checkpoint, TFLite INT8-mixed | .tflite size FP32 / INT8-mixed (MiB) | Phone latency FP32 / INT8-mixed (ms, 1 thread) |
|---|---|---|---|---|---|---|---|---|
{table}

* Accuracy: top-1 % on the official CIFAR-100 test set (10 000 images); the checkpoint was chosen on a held-out validation split (5 000 images), never on test.
  "3 seeds" = mean ± std of seeds 42 / 0 / 1 (the files here are the seed-42 models).
* INT8-mixed: regular convolutions and the classifier in int8, depthwise convolutions kept in float32 (plain INT8 lost 3.4 pp on w0.5 and broke MobileNetV3-Small).
* Latency: Samsung Galaxy F22 (Android 13, MediaTek Helio G80), TFLite `benchmark_model`, CPU + XNNPACK, 1 thread, median of 3 repetitions × 100 runs.
* The teacher has no INT8 file here (its INT8 version was not evaluated).

## Files per model

| File | Content |
|---|---|
| `model.safetensors` | PyTorch weights (state_dict of the best-by-validation checkpoint; optimizer state removed) |
| `config.json` | architecture, preprocessing, training recipe, results |
| `model_fp32.tflite` | TFLite float32 (identical predictions to PyTorch on all checked test images) |
| `model_int8_mixed.tflite` | TFLite INT8-mixed (students only) |

`classes.json` gives the 100 class names in label order.

## Input / preprocessing

Resize the shorter side to 147 px, centre-crop 128 × 128, scale to [0, 1], normalize with ImageNet mean (0.485, 0.456, 0.406)
and std (0.229, 0.224, 0.225). PyTorch expects NCHW; the `.tflite` files expect **NHWC float32 `[1, 128, 128, 3]`**. Output: 100 logits.

## Usage

PyTorch (model definitions from the GitHub repository, `src/models.py`):

```python
from safetensors.torch import load_file
from src.models import build_model

model = build_model("mobilenetv2", num_classes=100, width_mult=1.0, pretrained=False)  # see config.json of each folder
model.load_state_dict(load_file("models/mbv2_w1.0_kd/model.safetensors"))
model.eval()
```

TFLite:

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter   # or tf.lite.Interpreter

it = Interpreter(model_path="models/mbv2_w1.0_kd/model_int8_mixed.tflite")
it.allocate_tensors()
x = np.zeros((1, 128, 128, 3), np.float32)            # preprocessed image, NHWC
it.set_tensor(it.get_input_details()[0]["index"], x)
it.invoke()
logits = it.get_tensor(it.get_output_details()[0]["index"])
```

## Limitations

Research / course models trained on CIFAR-100 (32 × 32 images upsampled to 128 × 128); not intended for real-world use.
Single phone, CPU only. See the report in the GitHub repository (Section 9) for all limitations.

## References

1. G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531, 2015.
2. M. Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. arXiv:1801.04381.
"""
    (OUT / "README.md").write_text(card, encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"[hf] folder ready: {OUT}  ({total / 2**20:.1f} MiB, {sum(1 for p in OUT.rglob('*') if p.is_file())} files)")


if __name__ == "__main__":
    main()
