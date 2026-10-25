"""P25 — datasets and dataloaders (CIFAR-100; PlantVillage is Task 13).

Images are resized to `image_size` (default 128) so the same ImageNet-pretrained
backbones work for every dataset. ImageNet normalization is used because the
backbones are ImageNet-pretrained.

Splits (v2 protocol, no test-set selection)
-------------------------------------------
CIFAR-100's official 50k train set is split into 45k train + 5k val, stratified
(exactly `val_per_class` = 50 val images per class). The split depends ONLY on
`split_seed` (default 42), never on the training seed, so every run, seed and
the teacher see the same val images. The official 10k test set is untouched and
is evaluated once per run, on the best-by-val checkpoint.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_SEED_DEFAULT = 42
VAL_PER_CLASS_DEFAULT = 50


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


def stratified_split_indices(targets, val_per_class: int, split_seed: int):
    """Return (train_idx, val_idx): exactly `val_per_class` val images per class.

    Deterministic given (targets, val_per_class, split_seed); uses its own
    RandomState so it is independent of torch / python / numpy global seeds.
    """
    targets = np.asarray(targets)
    rng = np.random.RandomState(split_seed)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = rng.permutation(np.where(targets == c)[0])
        if len(idx) <= val_per_class:
            raise ValueError(f"class {c} has only {len(idx)} images (< val_per_class+1)")
        val_idx.extend(idx[:val_per_class].tolist())
        train_idx.extend(idx[val_per_class:].tolist())
    return sorted(train_idx), sorted(val_idx)


def _select(dataset, idx):
    """Copy of a torchvision CIFAR dataset keeping only `idx` (small to pickle for workers)."""
    out = copy.copy(dataset)
    out.data = dataset.data[idx]
    out.targets = [dataset.targets[i] for i in idx]
    return out


def _maybe_subset(dataset, smoke_test: bool, n: int):
    """For smoke tests, keep only the first `n` samples."""
    if not smoke_test:
        return dataset
    return Subset(dataset, list(range(min(n, len(dataset)))))


def _build_cifar100(cfg: dict, smoke_test: bool):
    root = cfg["data_root"]
    size = int(cfg["image_size"])
    Path(root).mkdir(parents=True, exist_ok=True)

    # two views of the same 50k images: augmented (train) and plain (val)
    full_train_view = datasets.CIFAR100(
        root=root, train=True, download=True,
        transform=build_transforms(size, train=True))
    full_eval_view = datasets.CIFAR100(
        root=root, train=True, download=True,
        transform=build_transforms(size, train=False))
    test_set = datasets.CIFAR100(
        root=root, train=False, download=True,
        transform=build_transforms(size, train=False))

    train_idx, val_idx = stratified_split_indices(
        full_train_view.targets,
        int(cfg.get("val_per_class", VAL_PER_CLASS_DEFAULT)),
        int(cfg.get("split_seed", SPLIT_SEED_DEFAULT)),
    )
    train_set = _select(full_train_view, train_idx)
    val_set = _select(full_eval_view, val_idx)

    train_set = _maybe_subset(train_set, smoke_test, 512)
    val_set = _maybe_subset(val_set, smoke_test, 256)
    test_set = _maybe_subset(test_set, smoke_test, 256)
    return train_set, val_set, test_set, 100


def _build_plantvillage(cfg: dict, smoke_test: bool):
    raise NotImplementedError(
        "PlantVillage needs a grouped train/val/test split (all photos of one leaf in "
        "one split). That is Task 13; the old random split leaked and was removed.")


def build_datasets(cfg: dict, smoke_test: bool = False):
    """Dispatch on cfg['dataset']. Returns (train_set, val_set, test_set, num_classes)."""
    name = str(cfg["dataset"]).lower()
    if name == "cifar100":
        return _build_cifar100(cfg, smoke_test)
    if name == "plantvillage":
        return _build_plantvillage(cfg, smoke_test)
    raise ValueError(f"Unknown dataset: {name!r} (use 'cifar100' or 'plantvillage')")


def build_dataloaders(cfg: dict, smoke_test: bool = False):
    """Return (train_loader, val_loader, test_loader, num_classes)."""
    train_set, val_set, test_set, num_classes = build_datasets(cfg, smoke_test)

    # On Windows, high worker counts + small smoke runs add overhead; keep it light.
    num_workers = 0 if smoke_test else int(cfg.get("num_workers", 4))
    pin = torch.cuda.is_available()
    batch_size = int(cfg["batch_size"])

    def make(ds, shuffle: bool, persistent: bool):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin, drop_last=False,
            persistent_workers=persistent and num_workers > 0,
        )

    train_loader = make(train_set, shuffle=True, persistent=True)
    val_loader = make(val_set, shuffle=False, persistent=True)
    # test is iterated only 1-2 times per run -> no persistent workers (saves RAM on Windows)
    test_loader = make(test_set, shuffle=False, persistent=False)
    return train_loader, val_loader, test_loader, num_classes
