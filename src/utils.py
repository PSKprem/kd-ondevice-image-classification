"""P25 — shared utilities: seeding, device, AMP, metrics, checkpoints, logging."""
from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_overrides(cfg: dict, overrides: dict) -> dict:
    """Override config values with any non-None CLI args (shallow merge)."""
    out = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster on fixed input sizes (our transforms are fixed-size)
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Tracks a running average (loss, accuracy, ...)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[float]:
    """Top-k accuracy (%) for the given values of k."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * 100.0 / batch_size).item())
    return res


# --------------------------------------------------------------------------- #
# LR schedule (cosine with linear warmup)
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Return a per-step LambdaLR implementing warmup + the chosen schedule."""
    epochs = int(cfg["epochs"])
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    sched = str(cfg.get("scheduler", "cosine")).lower()

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if sched == "none":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if sched == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if sched == "step":
            # drop by 10x at 50% and 75% of training
            if progress < 0.5:
                return 1.0
            if progress < 0.75:
                return 0.1
            return 0.01
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
@dataclass
class CheckpointState:
    epoch: int
    best_acc: float


def save_checkpoint(path: str | os.PathLike, model, optimizer, scheduler,
                    scaler, epoch: int, best_acc: float, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_acc": best_acc,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: str | os.PathLike, model, optimizer=None,
                    scheduler=None, scaler=None, map_location="cpu") -> CheckpointState:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return CheckpointState(epoch=int(ckpt.get("epoch", 0)),
                           best_acc=float(ckpt.get("best_acc", 0.0)))


# --------------------------------------------------------------------------- #
# CSV results logging (one row per finished run)
# --------------------------------------------------------------------------- #
RESULTS_FIELDS = [
    "run_name", "dataset", "model", "width_mult", "kd", "teacher_model",
    "temperature", "alpha", "epochs", "image_size", "seed",
    "params_m", "best_top1", "best_top5", "final_top1", "final_top5", "pretrained",
]


# v2 protocol (results/results_v2.csv): the best epoch is chosen on a held-out VAL
# split; the official TEST set is evaluated once on that best checkpoint.
#   best_val_top1/5 : val accuracy of the selected (best-by-val) checkpoint
#   test_top1/5     : test accuracy of that same best-by-val checkpoint (headline number)
#   final_test_top1/5: test accuracy of the LAST-epoch model (reported only, never used
#                      for selection)
RESULTS_FIELDS_V2 = [
    "run_name", "dataset", "model", "width_mult", "kd", "teacher_model",
    "temperature", "alpha", "epochs", "image_size", "seed",
    "optimizer", "lr", "weight_decay", "label_smoothing",
    "params_m", "n_train", "n_val", "best_epoch",
    "best_val_top1", "best_val_top5",
    "test_top1", "test_top5", "final_test_top1", "final_test_top5",
    "pretrained",
]


def append_result(csv_path: str | os.PathLike, row: dict[str, Any],
                  fields: list[str] | None = None) -> None:
    """Append one result row, writing a header if the file is new.

    `fields` defaults to the v1 schema. If the file already exists with a different
    header, raise instead of silently mixing schemas (old vs v2 results).
    """
    fields = list(fields) if fields is not None else RESULTS_FIELDS
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    if exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != fields:
            raise ValueError(f"{csv_path} has a different column schema than the one being "
                             "written; use a separate CSV file instead of mixing schemas.")
    safe = {k: row.get(k, "") for k in fields}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(safe)


def read_results(csv_path: str | os.PathLike) -> list[dict]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_params_millions(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
