"""P25 — evaluate a trained checkpoint (top-1 / top-5 accuracy).

Example
-------
    python -m src.evaluate --ckpt D:/kd-project/runs/<run>/best.pt --config configs/student.yaml
"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from .data import build_dataloaders
from .models import build_model
from .utils import (
    AverageMeter,
    accuracy,
    count_params_millions,
    get_device,
    load_checkpoint,
    load_config,
    merge_overrides,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P25 evaluator")
    p.add_argument("--ckpt", required=True, help="path to a checkpoint (best.pt/last.pt)")
    p.add_argument("--config", required=True, help="config describing data + model")
    p.add_argument("--width-mult", type=float, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="which split to score (default val; use test only for final numbers)")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    top1_m, top5_m = AverageMeter(), AverageMeter()
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        top1, top5 = accuracy(logits.float(), targets, topk=(1, 5))
        top1_m.update(top1, images.size(0))
        top5_m.update(top5, images.size(0))
    return top1_m.avg, top5_m.avg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, {"width_mult": args.width_mult})
    device = get_device()

    _, val_loader, test_loader, num_classes = build_dataloaders(cfg, smoke_test=args.smoke_test)
    loader = val_loader if args.split == "val" else test_loader
    model = build_model(cfg["model"], num_classes,
                        width_mult=float(cfg.get("width_mult", 1.0)),
                        pretrained=False)
    state = load_checkpoint(args.ckpt, model, map_location=device)
    model.to(device)

    top1, top5 = evaluate(model, loader, device)
    params_m = count_params_millions(model)
    print(f"[P25] ckpt={args.ckpt}")
    print(f"[P25] params={params_m:.2f}M  (trained {state.epoch} epochs, "
          f"best_val@1={state.best_acc:.2f})")
    print(f"[P25] eval[{args.split}]  top1={top1:.2f}  top5={top5:.2f}")


if __name__ == "__main__":
    main()
