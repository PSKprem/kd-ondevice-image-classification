# P25 — Knowledge Distillation for On-Device Image Classification

Compress a **ResNet-50** teacher into a **MobileNetV2** student that runs on a phone
with minimal accuracy loss, and study the **accuracy-vs-model-size Pareto curve** to
answer: *When does Knowledge Distillation help the most?*

- **Team 25 / Section B** — Shivang Dubey (su-24137), Moirangthem Joel Singh (su-24080), Rajiv Kumar (su-24109), Prem Shankar Kushawaha (su-24100)
- **Datasets:** CIFAR-100, PlantVillage
- **References:** Hinton et al. 2015 ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)), Sandler et al. CVPR 2018 ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381))

## KD loss (Hinton 2015)

```
L = alpha * T^2 * KL(softmax(z_s / T) || softmax(z_t / T)) + (1 - alpha) * CE(z_s, y)
```

## Setup

```powershell
# 1) Create + activate the virtual env (Python 3.12)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install PyTorch (CUDA build) FIRST, then the rest
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt` is the **training** env only. ONNX / TFLite export packages are in
`requirements-export.txt` and must go in a separate venv (or WSL / Colab): tensorflow needs
`numpy<2.1`, which conflicts with the numpy used for training.

Data and checkpoints live under `D:\kd-project\` (kept **out** of git — see `.gitignore`).
v2 runs (val-based protocol, below) write to `D:\kd-project\runs_v2\` and `results/results_v2.csv`.

## Evaluation protocol (v2)

CIFAR-100's 50k train set is split once into **45k train / 5k val** (stratified, 50 val images
per class, fixed `split_seed: 42` independent of the training seed). The best epoch is chosen on
**val**. After training, the official 10k **test** set is evaluated on `best.pt` (`test_top1`,
the headline number) and on the last-epoch model (`final_test_top1`, reported only).
`results/results_v2.csv` columns: `best_val_top1`, `test_top1`, `final_test_top1` (+ top-5, `best_epoch`, recipe).
The older `results/results.csv` (v1) chose the best epoch on the test set and is kept only as history.

## Usage

```powershell
# Train the teacher (ResNet-50)
python -m src.train --config configs/teacher.yaml

# Train a baseline student (no KD) at a given width
python -m src.train --config configs/student.yaml --width-mult 0.5

# Train a KD student (needs a trained teacher checkpoint)
python -m src.train --config configs/kd.yaml --teacher-ckpt D:/kd-project/runs_v2/teacher_resnet50_cifar100_v2/best.pt

# Evaluate a checkpoint (default split = val; pass --split test only for final numbers)
python -m src.evaluate --ckpt D:/kd-project/runs_v2/<run>/best.pt --config configs/student.yaml

# Quick 1-epoch smoke test (tiny subset; row goes to <run_dir>/smoke_results.csv, not the repo CSV)
python -m src.train --config configs/student.yaml --smoke-test
```

Resume any run after a `Ctrl+C` or crash:

```powershell
python -m src.train --config configs/teacher.yaml --resume
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/sweep.py` | Run the T x alpha grid (skips rows already in `results/results.csv`) |
| `scripts/export.py` | PyTorch -> ONNX -> TFLite (FP32 + INT8) |
| `scripts/benchmark.py` | Params / FLOPs / file size (+ on-device latency via adb) |
| `scripts/plot_results.py` | Pareto curve, T x alpha heatmap, KD-gain plots |

## Repository layout

```
configs/   teacher.yaml, student.yaml, kd.yaml
src/       data.py, models.py, kd_loss.py, train.py, evaluate.py, utils.py
scripts/   sweep.py, export.py, benchmark.py, plot_results.py
results/   results.csv, figures/
notebooks/ analysis.ipynb (optional)
report/    report.pdf
```

See `project.md` and `task.md` for the full brief and night-by-night plan.
