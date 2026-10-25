"""P25 diagnostic — why does INT8 hurt some students (w0.5, MBV3-Small) but not others (w1.0, w1.4)?

Pure PyTorch (CPU), no TFLite: fake-quantize the CONV/LINEAR WEIGHTS to int8 (symmetric, per output channel) and measure
test accuracy on the first N CIFAR-100 test images, in two variants:
  - unfolded : quantize the weights as they are in the trained model
  - folded   : first fold BatchNorm into the preceding conv (what TFLite does), then quantize
If "folded" loses accuracy but "unfolded" does not, BatchNorm folding makes the weight ranges quantization-unfriendly.
If neither loses accuracy, the loss comes from something else in the TFLite path (e.g. activation quantization).

Usage (training venv, from repo root):
    python scripts/diag_quant.py [--n 2000]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import build_datasets  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_checkpoint, load_config  # noqa: E402

RUNS = Path("D:/kd-project/runs_v2")
MODELS = [  # (label, run, model name, width)
    ("w0.5", "kd_cifar100_w0.5_v2_adamw1e-3_T2_a0.9", "mobilenetv2", 0.5),
    ("w1.0", "kd_cifar100_w1.0_v2_sgd0.01_T2_a0.9", "mobilenetv2", 1.0),
    ("w1.4", "kd_cifar100_w1.4_v2_sgd0.01_T2_a0.9", "mobilenetv2", 1.4),
    ("mbv3s", "kd_cifar100_mbv3s_v2_adamw1e-3_T2_a0.9", "mobilenetv3_small", 1.0),
]


def fold_bn(model: nn.Module) -> int:
    """Fold each BatchNorm2d into the Conv2d registered right before it (flat module order). Returns #folded."""
    mods = [(n, m) for n, m in model.named_modules() if not list(m.children()) or isinstance(m, nn.BatchNorm2d)]
    folded = 0
    for (_, a), (_, b) in zip(mods, mods[1:]):
        if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d) and a.out_channels == b.num_features:
            std = torch.sqrt(b.running_var + b.eps)
            scale = (b.weight / std) if b.weight is not None else 1.0 / std
            shape = [-1] + [1] * (a.weight.dim() - 1)
            a.weight.data = a.weight.data * scale.reshape(shape)
            bias = a.bias.data if a.bias is not None else torch.zeros(a.out_channels)
            new_bias = (bias - b.running_mean) * scale + (b.bias if b.bias is not None else 0.0)
            a.bias = nn.Parameter(new_bias.detach())
            # make the BN an identity (its activation, if any, is kept)
            b.running_mean.zero_()
            b.running_var.fill_(1.0 - b.eps)  # sqrt(var + eps) == 1
            if b.weight is not None:
                b.weight.data.fill_(1.0)
                b.bias.data.zero_()
            folded += 1
    return folded


def layer_group(m: nn.Module, first_conv: nn.Module) -> str | None:
    if isinstance(m, nn.Linear):
        return "classifier"
    if isinstance(m, nn.Conv2d):
        if m is first_conv:
            return "first conv"
        if m.groups > 1 and m.groups == m.in_channels:
            return "depthwise"
        return "pointwise" if m.kernel_size == (1, 1) else "other conv"
    return None


def quantize_weights_(model: nn.Module, only: str | None = None) -> None:
    """Symmetric per-output-channel int8 fake quantization of every Conv2d / Linear weight (or only one layer group)."""
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and (only is None or layer_group(m, first_conv) == only):
            w = m.weight.data
            flat = w.reshape(w.shape[0], -1)
            scale = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            q = torch.round(flat / scale[:, None]).clamp(-127, 127) * scale[:, None]
            m.weight.data = q.reshape(w.shape)


@torch.no_grad()
def accuracy(model: nn.Module, test_set, n: int) -> float:
    model.eval()
    correct = 0
    for i in range(0, n, 100):
        xs, ys = zip(*[test_set[j] for j in range(i, min(i + 100, n))])
        correct += int((model(torch.stack(xs)).argmax(1) == torch.tensor(ys)).sum())
    return 100.0 * correct / n


def main() -> None:
    ap = argparse.ArgumentParser(description="P25 INT8 weight-quantization diagnostic")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--groups", action="store_true", help="also quantize one layer group at a time")
    args = ap.parse_args()
    cfg = load_config("configs/student.yaml")
    _, _, test_set, num_classes = build_datasets(cfg)
    torch.set_num_threads(4)
    print(f"{'model':7s} {'FP32':>7s} {'int8 unfolded':>14s} {'int8 folded':>12s} {'#BN folded':>11s}   (test top-1 on first {args.n} images)")
    for label, run, name, width in MODELS:
        base = build_model(name, num_classes, width_mult=width, pretrained=False)
        load_checkpoint(str(RUNS / run / "best.pt"), base, map_location="cpu")
        fp = accuracy(base, test_set, args.n)
        m1 = copy.deepcopy(base)
        quantize_weights_(m1)
        a1 = accuracy(m1, test_set, args.n)
        m2 = copy.deepcopy(base)
        nf = fold_bn(m2)
        chk = accuracy(m2, test_set, 300) - accuracy(base, test_set, 300)  # folding alone must not change accuracy
        quantize_weights_(m2)
        a2 = accuracy(m2, test_set, args.n)
        print(f"{label:7s} {fp:7.2f} {a1:14.2f} {a2:12.2f} {nf:11d}   (fold sanity: acc change before quantizing = {chk:+.2f})", flush=True)
        if args.groups:
            parts = []
            for g in ("first conv", "depthwise", "pointwise", "classifier"):
                mg = copy.deepcopy(base)
                quantize_weights_(mg, only=g)
                parts.append(f"{g}: {accuracy(mg, test_set, args.n) - fp:+.2f}")
            print("          int8 on ONE layer group only (change vs FP32, pp):  " + " | ".join(parts), flush=True)


if __name__ == "__main__":
    main()
