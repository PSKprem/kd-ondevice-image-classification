"""P25 — training entry point (baseline or Knowledge Distillation).

Examples
--------
    python -m src.train --config configs/teacher.yaml
    python -m src.train --config configs/student.yaml --width-mult 0.5
    python -m src.train --config configs/kd.yaml --teacher-ckpt D:/kd-project/runs/teacher.../best.pt
    python -m src.train --config configs/student.yaml --smoke-test
    python -m src.train --config configs/teacher.yaml --resume
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import build_dataloaders
from .kd_loss import KDLoss
from .models import build_model
from .utils import (
    RESULTS_FIELDS_V2,
    AverageMeter,
    accuracy,
    append_result,
    build_scheduler,
    count_params_millions,
    get_device,
    load_checkpoint,
    load_config,
    merge_overrides,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P25 KD trainer")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--width-mult", type=float, default=None, help="override student width")
    p.add_argument("--model", type=str, default=None,
                   help="override model (resnet50 | mobilenetv2 | mobilenetv3_small)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="random init instead of ImageNet weights (from-scratch experiment)")
    p.add_argument("--teacher-ckpt", type=str, default=None, help="teacher checkpoint for KD")
    p.add_argument("--temperature", type=float, default=None, help="override KD temperature T")
    p.add_argument("--alpha", type=float, default=None, help="override KD alpha")
    p.add_argument("--epochs", type=int, default=None, help="override epochs")
    p.add_argument("--label-smoothing", type=float, default=None, help="override label smoothing")
    p.add_argument("--lr", type=float, default=None, help="override learning rate")
    p.add_argument("--optimizer", type=str, default=None, help="override optimizer (sgd | adamw)")
    p.add_argument("--weight-decay", type=float, default=None, help="override weight decay")
    p.add_argument("--warmup-epochs", type=int, default=None, help="override warmup epochs")
    p.add_argument("--results-csv", type=str, default=None, help="override results CSV path")
    p.add_argument("--seed", type=int, default=None, help="override random seed")
    p.add_argument("--dataset", type=str, default=None, help="override dataset (cifar100 | plantvillage)")
    p.add_argument("--train-fraction", type=float, default=None,
                   help="low-data study: use this fraction of the CIFAR-100 train split (stratified, nested)")
    p.add_argument("--plantvillage-dir", type=str, default=None,
                   help="override PlantVillage 'color' folder (one sub-folder per class)")
    p.add_argument("--run-name", type=str, default=None, help="override run_name")
    p.add_argument("--resume", action="store_true", help="resume from last.pt in the run dir")
    p.add_argument("--smoke-test", action="store_true", help="tiny 1-epoch end-to-end test")
    return p.parse_args()


def build_criterion(cfg: dict, device: torch.device):
    """Return (criterion, is_kd). KD uses KDLoss, baseline uses CE."""
    if cfg.get("kd", False):
        crit = KDLoss(
            temperature=float(cfg.get("temperature", 4.0)),
            alpha=float(cfg.get("alpha", 0.5)),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        )
        return crit, True
    crit = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    return crit.to(device), False


def load_teacher(cfg: dict, num_classes: int, device: torch.device) -> nn.Module:
    ckpt_path = cfg.get("teacher_ckpt")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"KD requires a trained teacher checkpoint; not found: {ckpt_path!r}. "
            "Train the teacher first or pass --teacher-ckpt."
        )
    teacher = build_model(cfg.get("teacher_model", "resnet50"), num_classes,
                          width_mult=1.0, pretrained=False)
    load_checkpoint(ckpt_path, teacher, map_location=device)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def run_epoch(model, loader, criterion, is_kd, teacher, optimizer, scheduler,
              scaler, device, amp, train: bool):
    model.train(train)
    loss_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()
    desc = "train" if train else "eval"
    autocast_dev = "cuda" if device.type == "cuda" else "cpu"

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.size(0)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=autocast_dev, enabled=amp):
                logits = model(images)
                if is_kd:
                    with torch.no_grad():
                        t_logits = teacher(images)
                    out = criterion(logits, t_logits, targets)
                    loss = out["loss"]
                else:
                    loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        top1, top5 = accuracy(logits.float(), targets, topk=(1, 5))
        loss_m.update(loss.item(), bs)
        top1_m.update(top1, bs)
        top5_m.update(top5, bs)

    return loss_m.avg, top1_m.avg, top5_m.avg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, {
        "width_mult": args.width_mult,
        "teacher_ckpt": args.teacher_ckpt,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "run_name": args.run_name,
        "model": args.model,
        "label_smoothing": args.label_smoothing,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "results_csv": args.results_csv,
        "seed": args.seed,
        "dataset": args.dataset,
        "train_fraction": args.train_fraction,
        "plantvillage_dir": args.plantvillage_dir,
    })
    if args.no_pretrained:
        cfg["pretrained"] = False

    smoke = args.smoke_test
    if smoke:
        cfg["epochs"] = 1
        cfg["run_name"] = cfg.get("run_name", "run") + "_smoke"

    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    print(f"[P25] device={device} amp={amp} run={cfg.get('run_name')}")
    print(f"[P25] recipe: opt={cfg.get('optimizer')} lr={cfg.get('lr')} "
          f"wd={cfg.get('weight_decay')} warmup={cfg.get('warmup_epochs')} "
          f"label_smoothing={cfg.get('label_smoothing')} epochs={cfg.get('epochs')}")

    # ---- data (train / val / test; test is touched only after training) ----
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        cfg, smoke_test=smoke)
    n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)
    print(f"[P25] dataset={cfg['dataset']} classes={num_classes} "
          f"train={n_train} val={n_val} test={len(test_loader.dataset)}")

    # ---- model ----
    model = build_model(cfg["model"], num_classes,
                        width_mult=float(cfg.get("width_mult", 1.0)),
                        pretrained=bool(cfg.get("pretrained", True)))
    model.to(device)
    params_m = count_params_millions(model)
    print(f"[P25] model={cfg['model']} width={cfg.get('width_mult', 1.0)} "
          f"params={params_m:.2f}M")

    # ---- loss (baseline CE or KD) ----
    criterion, is_kd = build_criterion(cfg, device)
    teacher = load_teacher(cfg, num_classes, device) if is_kd else None
    if is_kd:
        print(f"[P25] KD on: T={cfg.get('temperature')} alpha={cfg.get('alpha')} "
              f"teacher={cfg.get('teacher_model')}")

    # ---- optimizer / scheduler / scaler ----
    opt_name = str(cfg.get("optimizer", "sgd")).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=float(cfg["lr"]),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=float(cfg.get("weight_decay", 0.0)), nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(cfg["lr"]),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # ---- run dir + resume ----
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"

    start_epoch, best_acc = 0, 0.0
    if not args.resume and not smoke and (last_path.exists() or best_path.exists()):
        raise SystemExit(f"[P25] {run_dir} already has checkpoints. Use --resume to continue "
                         "it, or pick a new --run-name (refusing to overwrite).")
    if args.resume and last_path.exists():
        state = load_checkpoint(last_path, model, optimizer, scheduler, scaler,
                                map_location=device)
        start_epoch, best_acc = state.epoch, state.best_acc
        print(f"[P25] resumed from epoch {start_epoch} (best_acc={best_acc:.2f})")

    # ---- train loop: the best epoch is selected on VAL only ----
    epochs = int(cfg["epochs"])
    for epoch in range(start_epoch, epochs):
        tr_loss, tr_top1, _ = run_epoch(
            model, train_loader, criterion, is_kd, teacher, optimizer,
            scheduler, scaler, device, amp, train=True)
        va_loss, va_top1, va_top5 = run_epoch(
            model, val_loader, criterion, is_kd, teacher, None,
            None, scaler, device, amp, train=False)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[P25] epoch {epoch+1}/{epochs} lr={lr_now:.5f} "
              f"train_loss={tr_loss:.3f} train@1={tr_top1:.2f} "
              f"val_loss={va_loss:.3f} val@1={va_top1:.2f} val@5={va_top5:.2f}")

        is_best = va_top1 > best_acc or not best_path.exists()
        best_acc = max(best_acc, va_top1)
        if (epoch + 1) % int(cfg.get("save_every", 1)) == 0 or epoch + 1 == epochs:
            save_checkpoint(last_path, model, optimizer, scheduler, scaler,
                            epoch + 1, best_acc, cfg)
        if is_best:
            save_checkpoint(best_path, model, optimizer, scheduler, scaler,
                            epoch + 1, best_acc, cfg)

    # ---- test set: touched only here, after training is over ----
    # (1) last-epoch model -> final_test_*  (reported only, never used for selection)
    _, final_test_top1, final_test_top5 = run_epoch(
        model, test_loader, criterion, is_kd, teacher, None, None, scaler, device, amp,
        train=False)
    # (2) best-by-val checkpoint -> test_*  (the headline number, evaluated once)
    best_state = load_checkpoint(best_path, model, map_location=device)
    _, val_chk_top1, best_val_top5 = run_epoch(
        model, val_loader, criterion, is_kd, teacher, None, None, scaler, device, amp,
        train=False)
    if abs(val_chk_top1 - best_state.best_acc) > 0.1:
        print(f"[P25] WARNING: val re-check {val_chk_top1:.2f} differs from saved "
              f"best {best_state.best_acc:.2f}")
    _, test_top1, test_top5 = run_epoch(
        model, test_loader, criterion, is_kd, teacher, None, None, scaler, device, amp,
        train=False)

    print(f"[P25] done. best epoch={best_state.epoch} best_val@1={best_state.best_acc:.2f}  "
          f"test@1(best)={test_top1:.2f}  final_test@1(last epoch)={final_test_top1:.2f}  "
          f"(checkpoints in {run_dir})")

    # ---- log one result row (smoke tests go to a throw-away CSV in the run dir) ----
    csv_path = run_dir / "smoke_results.csv" if smoke else cfg["results_csv"]
    append_result(csv_path, {
        "run_name": cfg["run_name"], "dataset": cfg["dataset"],
        "model": cfg["model"], "width_mult": cfg.get("width_mult", 1.0),
        "kd": int(bool(cfg.get("kd", False))),
        "teacher_model": cfg.get("teacher_model", "") if cfg.get("kd", False) else "",
        "temperature": cfg.get("temperature", "") if cfg.get("kd", False) else "",
        "alpha": cfg.get("alpha", "") if cfg.get("kd", False) else "",
        "epochs": epochs, "image_size": cfg["image_size"],
        "seed": cfg.get("seed", 42),
        "optimizer": cfg.get("optimizer", ""), "lr": cfg.get("lr", ""),
        "weight_decay": cfg.get("weight_decay", ""),
        "label_smoothing": cfg.get("label_smoothing", ""),
        "params_m": round(params_m, 4), "n_train": n_train, "n_val": n_val,
        "best_epoch": best_state.epoch,
        "best_val_top1": round(best_state.best_acc, 4),
        "best_val_top5": round(best_val_top5, 4),
        "test_top1": round(test_top1, 4), "test_top5": round(test_top5, 4),
        "final_test_top1": round(final_test_top1, 4),
        "final_test_top5": round(final_test_top5, 4),
        "pretrained": int(bool(cfg.get("pretrained", True))),
    }, fields=RESULTS_FIELDS_V2)
    print(f"[P25] appended result to {csv_path}")


if __name__ == "__main__":
    main()
