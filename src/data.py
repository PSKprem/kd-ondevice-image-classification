"""P25 — datasets and dataloaders (CIFAR-100 + PlantVillage).

Images are resized to `image_size` (default 128) so the same ImageNet-pretrained
backbones work for both datasets. ImageNet normalization is used because the
backbones are ImageNet-pretrained.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool):
    """Return train/eval transform pipelines."""
    if train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _maybe_subset(dataset, smoke_test: bool, n: int = 512):
    """For smoke tests, keep only the first `n` samples."""
    if not smoke_test:
        return dataset
    n = min(n, len(dataset))
    return Subset(dataset, list(range(n)))


def _build_cifar100(cfg: dict, smoke_test: bool):
    root = cfg["data_root"]
    size = int(cfg["image_size"])
    Path(root).mkdir(parents=True, exist_ok=True)

    train_set = datasets.CIFAR100(
        root=root, train=True, download=True,
        transform=build_transforms(size, train=True),
    )
    test_set = datasets.CIFAR100(
        root=root, train=False, download=True,
        transform=build_transforms(size, train=False),
    )
    num_classes = 100
    train_set = _maybe_subset(train_set, smoke_test)
    test_set = _maybe_subset(test_set, smoke_test, n=256)
    return train_set, test_set, num_classes


def _build_plantvillage(cfg: dict, smoke_test: bool):
    """PlantVillage as an ImageFolder. Expects either:
        <plantvillage_dir>/train/<class>/*.jpg and .../val/<class>/*.jpg
    or a single <plantvillage_dir>/<class>/*.jpg (auto-split by val_split).
    """
    root = Path(cfg["plantvillage_dir"])
    size = int(cfg["image_size"])
    if not root.exists():
        raise FileNotFoundError(
            f"PlantVillage directory not found: {root}. "
            "Download it (kaggle) and point plantvillage_dir at it."
        )

    train_dir, val_dir = root / "train", root / "val"
    if train_dir.exists() and val_dir.exists():
        train_set = datasets.ImageFolder(train_dir, build_transforms(size, True))
        test_set = datasets.ImageFolder(val_dir, build_transforms(size, False))
        num_classes = len(train_set.classes)
    else:
        full_train = datasets.ImageFolder(root, build_transforms(size, True))
        full_eval = datasets.ImageFolder(root, build_transforms(size, False))
        num_classes = len(full_train.classes)
        val_frac = float(cfg.get("val_split", 0.1))
        n_val = int(len(full_train) * val_frac)
        n_train = len(full_train) - n_val
        g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        train_idx, val_idx = random_split(range(len(full_train)), [n_train, n_val], generator=g)
        train_set = Subset(full_train, list(train_idx))
        test_set = Subset(full_eval, list(val_idx))

    train_set = _maybe_subset(train_set, smoke_test)
    test_set = _maybe_subset(test_set, smoke_test, n=256)
    return train_set, test_set, num_classes


def build_datasets(cfg: dict, smoke_test: bool = False):
    """Dispatch on cfg['dataset']. Returns (train_set, test_set, num_classes)."""
    name = str(cfg["dataset"]).lower()
    if name == "cifar100":
        return _build_cifar100(cfg, smoke_test)
    if name == "plantvillage":
        return _build_plantvillage(cfg, smoke_test)
    raise ValueError(f"Unknown dataset: {name!r} (use 'cifar100' or 'plantvillage')")


def build_dataloaders(cfg: dict, smoke_test: bool = False):
    """Return (train_loader, test_loader, num_classes)."""
    train_set, test_set, num_classes = build_datasets(cfg, smoke_test)

    # On Windows, high worker counts + small smoke runs add overhead; keep it light.
    num_workers = 0 if smoke_test else int(cfg.get("num_workers", 4))
    pin = torch.cuda.is_available()
    batch_size = int(cfg["batch_size"])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, num_classes
