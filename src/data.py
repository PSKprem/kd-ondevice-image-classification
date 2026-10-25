"""P25 — datasets and dataloaders (CIFAR-100 and PlantVillage).

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

PlantVillage (color, 38 classes, 54,305 images) has no leaf id, so it is split per class into blocks of
`block_size` (20) consecutive image ids; whole blocks go to train / val / test (~70/10/20, `split_seed`).
This is an approximation that reduces (not removes) near-duplicate leakage between splits.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_SEED_DEFAULT = 42
VAL_PER_CLASS_DEFAULT = 50
# PlantVillage: per class, blocks of 20 consecutive ids -> ~70% train / 10% val / 20% test
PV_BLOCK_SIZE_DEFAULT = 20
PV_VAL_FRAC_DEFAULT = 0.1
PV_TEST_FRAC_DEFAULT = 0.2


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


def _plantvillage_key(filename: str):
    """Sort key (prefix, number) from names like '<uuid>___FREC_Scab 2907.JPG' or 'RS_Rust 1563.JPG'.

    The number is a per-image id; consecutive ids tend to be shots of the same leaf
    (adjacent ids are measurably more similar than random same-class images), but the
    dataset has NO explicit leaf id, so grouping by id blocks is only an approximation.
    """
    stem = Path(filename).stem.split("___")[-1]
    m = re.match(r"(.*?)\s*(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else (stem, -1)


def block_split_indices(class_indices: dict, block_size: int, val_frac: float,
                        test_frac: float, split_seed: int):
    """Block split. `class_indices`: class -> list of sample indices ALREADY sorted by id.

    Each class is cut into blocks of `block_size` consecutive ids; whole blocks go to
    train / val / test (random per class, own RandomState(split_seed)). Returns
    (train_idx, val_idx, test_idx). Deterministic, independent of the training seed.
    """
    rng = np.random.RandomState(split_seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in sorted(class_indices):
        idx = class_indices[c]
        blocks = [idx[i:i + block_size] for i in range(0, len(idx), block_size)]
        order = rng.permutation(len(blocks))
        n_test = max(1, round(test_frac * len(blocks)))
        n_val = max(1, round(val_frac * len(blocks)))
        if len(blocks) < n_test + n_val + 1:
            raise ValueError(f"class {c}: only {len(blocks)} blocks, too few for a 3-way split")
        for k, b in enumerate(order):
            target = test_idx if k < n_test else val_idx if k < n_test + n_val else train_idx
            target.extend(blocks[b])
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


class _PathDataset(torch.utils.data.Dataset):
    """(image path, label) list + transform; light to pickle for DataLoader workers."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label


def _build_plantvillage(cfg: dict, smoke_test: bool):
    root = Path(cfg.get("plantvillage_dir", ""))
    if not root.is_dir():
        raise FileNotFoundError(
            f"PlantVillage folder not found: {root}. Set `plantvillage_dir` in the config to the "
            "'color' folder (one sub-folder per class).")
    size = int(cfg["image_size"])
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    samples, class_indices = [], {}
    for ci, cname in enumerate(classes):
        files = sorted((f for f in os.listdir(root / cname)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))), key=_plantvillage_key)
        class_indices[ci] = list(range(len(samples), len(samples) + len(files)))
        samples.extend((str(root / cname / f), ci) for f in files)

    train_idx, val_idx, test_idx = block_split_indices(
        class_indices,
        int(cfg.get("block_size", PV_BLOCK_SIZE_DEFAULT)),
        float(cfg.get("val_frac", PV_VAL_FRAC_DEFAULT)),
        float(cfg.get("test_frac", PV_TEST_FRAC_DEFAULT)),
        int(cfg.get("split_seed", SPLIT_SEED_DEFAULT)),
    )
    train_t, eval_t = build_transforms(size, True), build_transforms(size, False)
    train_set = _PathDataset([samples[i] for i in train_idx], train_t)
    val_set = _PathDataset([samples[i] for i in val_idx], eval_t)
    test_set = _PathDataset([samples[i] for i in test_idx], eval_t)

    train_set = _maybe_subset(train_set, smoke_test, 512)
    val_set = _maybe_subset(val_set, smoke_test, 256)
    test_set = _maybe_subset(test_set, smoke_test, 256)
    return train_set, val_set, test_set, len(classes)


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
