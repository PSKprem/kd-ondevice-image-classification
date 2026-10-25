# P25 — Knowledge Distillation for On-Device Image Classification

Compress a **ResNet-50** teacher into a small **MobileNet** student that runs on a phone
with minimal accuracy loss, and study the **accuracy-vs-model-size Pareto curve** to
answer: *When does Knowledge Distillation help the most?*

- **Team 25 / Section B** — Shivang Dubey (su-24137), Moirangthem Joel Singh (su-24080), Rajiv Kumar (su-24109), Prem Shankar Kushawaha (su-24100). The implementation and experiments in this repository were carried out by a single member.
- **Datasets:** CIFAR-100 (main), PlantVillage (smaller second study)
- **References:** Hinton et al. 2015 ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)), Sandler et al. CVPR 2018 ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381))
- Project plan and live status: [`task.md`](task.md) · brief: [`project.md`](project.md)

## KD loss (Hinton 2015)

```
L = alpha * T^2 * KL(softmax(z_s / T) || softmax(z_t / T)) + (1 - alpha) * CE(z_s, y)
```

## Setup

Three separate environments (they have conflicting numpy / TensorFlow requirements):

```powershell
# 1) Training env (Python 3.12): PyTorch CUDA build FIRST, then the rest
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2) Export env (ONNX / TFLite), verified versions in requirements-export.txt
py -3.12 -m venv D:\kd-project\.venv-export
D:\kd-project\.venv-export\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-export.txt

# 3) (optional) Kaggle CLI, only to download PlantVillage
py -3.12 -m venv D:\kd-project\.venv-kaggle
D:\kd-project\.venv-kaggle\Scripts\pip install kaggle
```

Data, checkpoints, exports and logs live under `D:\kd-project\` (kept **out** of git — see `.gitignore`).
Runs write to `D:\kd-project\runs_v2\` and `results/results_v2.csv`.

### Data

- **CIFAR-100** downloads automatically into `D:\kd-project\data`.
- **PlantVillage** (Kaggle `abdallahalidev/plantvillage-dataset`, 38 classes, 54,305 colour images). Needs a Kaggle API key in `C:\Users\<you>\.kaggle\kaggle.json` (never commit it):

```powershell
D:\kd-project\.venv-kaggle\Scripts\kaggle datasets download -d abdallahalidev/plantvillage-dataset -p D:\kd-project\data\plantvillage_raw
# extract only the "plantvillage dataset/color" folder; configs point to it via `plantvillage_dir`
```

## Evaluation protocol (v2)

**CIFAR-100:** the 50k train set is split once into **45k train / 5k val** (stratified, 50 val images
per class, fixed `split_seed: 42` independent of the training seed). The best epoch is chosen on
**val**. After training, the official 10k **test** set is evaluated on `best.pt` (`test_top1`,
the headline number) and on the last-epoch model (`final_test_top1`, reported only).

**PlantVillage:** there is no leaf id in the file names, so each class is cut into blocks of 20
consecutive image ids and whole blocks go to train / val / test (~70 / 10 / 20 %, 37,994 / 5,425 / 10,886 images).
This reduces — but cannot fully remove — near-duplicate leakage between splits (a known limitation).

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

# Same on PlantVillage (give the run a unique name; the configs default to the CIFAR-100 name)
python -m src.train --config configs/student.yaml --dataset plantvillage --run-name student_mbv2_pv_w0.5 --width-mult 0.5

# Evaluate a checkpoint (default split = val; pass --split test only for final numbers)
python -m src.evaluate --ckpt D:/kd-project/runs_v2/<run>/best.pt --config configs/student.yaml

# Quick 1-epoch smoke test (tiny subset; row goes to <run_dir>/smoke_results.csv, not the repo CSV)
python -m src.train --config configs/student.yaml --smoke-test
```

Resume any run after a `Ctrl+C` or crash: add `--resume`. The queue scripts below are resumable too (finished runs are skipped).

## Experiment queues (v2)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\queue_v2_night1.ps1   # teacher + 8 baselines + 12-run T x alpha grid (w1.0)
powershell -ExecutionPolicy Bypass -File scripts\queue_v2_exp4.ps1     # KD (T=2, alpha=0.9) on every student + alpha probe
python scripts/plot_results.py                                          # figures + results/summary_table_v2.csv
```

## Export and on-device pipeline

Run in the export env (`D:\kd-project\.venv-export`, add its `Scripts` folder to `PATH`):

```powershell
python scripts/export.py --ckpt <run>/best.pt --config configs/student.yaml --width-mult 1.0 --out D:/kd-project/export/<name> --onnx-only
onnx2tf -i D:/kd-project/export/<name>/model.onnx -o D:/kd-project/export/<name>/tf -b 1        # -> model_float32.tflite
python scripts/quantize_tflite.py --fp32 .../tf/model_float32.tflite --out .../model_int8_dynamic.tflite --mode dynamic
python scripts/quantize_tflite.py --fp32 .../tf/model_float32.tflite --out .../model_int8_static.tflite --mode static --calib <calib.npy>
python scripts/eval_tflite.py --tflite <file.tflite> --config configs/student.yaml --ckpt <run>/best.pt --width-mult 1.0 --n 0
python scripts/benchmark.py ...                                                                  # params / FLOPs / size (+ adb latency)
```

Phone latency is measured with TFLite `benchmark_model` through `adb` (see `scripts/benchmark.py`). See `requirements-export.txt` for what did and did not work in the TFLite toolchain.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/queue_v2_night1.py` / `.ps1` | v2: teacher, 8 baselines (4 students x 2 recipes), full T x alpha grid on w1.0 |
| `scripts/queue_v2_exp4.py` / `.ps1` | v2: KD with one common (T, alpha) on all students; `--seeds` extra seeds, `--probe` alpha sensitivity |
| `scripts/plot_results.py` | Pareto, KD gain vs size, T x alpha heatmap, alpha probe, summary table (v2 CSV) |
| `scripts/export.py` | PyTorch checkpoint -> ONNX (TFLite step: see above) |
| `scripts/quantize_tflite.py` | FP32 `.tflite` -> INT8 (dynamic or static) with ai-edge-quantizer |
| `scripts/eval_tflite.py` | Test accuracy of a `.tflite` and agreement with the PyTorch model |
| `scripts/benchmark.py` | Params / FLOPs / file size (+ on-device latency via adb) |
| `scripts/queue_night3.*`, `queue_exp4.py`, `queue_seeds.py`, `sweep*.{py,ps1}`, `overnight_night1.ps1`, `pretrained_baselines.ps1` | v1 history (test-selected protocol); not used for headline results |

## Results so far (CIFAR-100, v2, test top-1; mean over seeds where n > 1)

| Model | Params (M) | Seeds | Baseline | KD (T=2, alpha=0.9) | Gain (pp) |
|---|---|---|---|---|---|
| ResNet-50 (teacher) | 23.71 | 1 | 85.46 | — | — |
| MobileNetV2 w0.5 | 0.82 | 3 | 74.71 ± 0.16 | 75.17 ± 0.15 | +0.46 |
| MobileNetV3-Small | 1.62 | 1 | 78.34 | 80.02 | +1.68 |
| MobileNetV2 w1.0 | 2.35 | 3 | 80.82 ± 0.07 | 81.63 ± 0.18 | +0.81 |
| MobileNetV2 w1.4 | 4.50 | 1 | 82.36 | 83.61 | +1.25 |

Per-seed paired gains (seeds 42 / 0 / 1): w0.5 +0.45 / +0.44 / +0.49, w1.0 +0.85 / +0.89 / +0.68, i.e. small but
positive for every seed. w1.4 and MobileNetV3-Small are still single-seed. The gain grows with width inside
MobileNetV2, but optimizer (AdamW for w0.5 / MBV3-S, SGD for w1.0 / w1.4, each chosen on val) and architecture are
mixed with size, so size alone cannot be credited. The final tables and figures are produced by
`scripts/plot_results.py` (`results/summary_table_v2.csv`, `results/figures/`). On-device latency results are still
to come (see `task.md`).

### Exported TFLite models (CIFAR-100 test set; INT8 = dynamic-range, full 10k images)

| Model | FP32 .tflite (MiB) | INT8 .tflite (MiB) | PyTorch FP32 KD / baseline | INT8 KD / baseline |
|---|---|---|---|---|
| MobileNetV2 w0.5 | 3.12 | 0.94 | 75.03 / 74.58 | 71.63 / 71.75 |
| MobileNetV3-Small | 6.22 | 1.75 | 80.02 / 78.34 | 13.14 / 9.76 (broken) |
| MobileNetV2 w1.0 | 8.95 | 2.52 | 81.71 / 80.86 | 81.78 / 80.59 |
| MobileNetV2 w1.4 | 17.10 | 4.65 | 83.61 / 82.36 | 83.23 / 81.97 |
| ResNet-50 (teacher) | 90.39 | — | 85.46 (FP32) | — |

FP32 TFLite predictions are identical to PyTorch (100% agreement on all models). Dynamic INT8 is accurate for
w1.0 / w1.4 (and the KD advantage of about +1.2 pp survives quantization), loses 3.4 pp on w0.5 and breaks MobileNetV3-Small.
On 2000 test images, weight-only INT8 keeps MBV3-S usable (79.65 vs 80.30 FP32) but still costs w0.5 about 2.9 pp
(72.85 vs 75.75); static INT8 was worse. The reason is not established. Latency on the phone is still to come.

### Low-data study (CIFAR-100, pretrained, test top-1, 20 epochs, single seed, KD T=2 alpha=0.9)

| Student | Train data | Baseline | KD | Gain (pp) |
|---|---|---|---|---|
| MobileNetV2 w1.0 (SGD) | 10% | 65.81 | 66.93 | +1.12 |
| MobileNetV2 w1.0 (SGD) | 25% | 72.72 | 74.75 | +2.03 |
| MobileNetV2 w1.0 (SGD) | 100% | 80.86 | 81.71 | +0.85 |
| MobileNetV2 w0.5 (AdamW) | 10% | 56.44 | 56.22 | -0.22 |
| MobileNetV2 w0.5 (AdamW) | 25% | 64.87 | 64.97 | +0.10 |
| MobileNetV2 w0.5 (AdamW) | 100% | 74.58 | 75.03 | +0.45 |

For w1.0, KD helps more with less data (+2.03 at 25% vs +0.85 at 100%), but not monotonically (10% is lower). For the
tiny w0.5 student KD does not help when data is scarce (-0.22 at 10%). So "KD helps most with little data" is not a
general rule here; the two students also differ in optimizer. The teacher was trained on the full data.

### Optimizer check (CIFAR-100, pretrained, seed 42, test top-1 KD gain in pp, KD T=2 alpha=0.9)

| Student | SGD 0.01 baseline -> KD (gain) | AdamW 1e-3 baseline -> KD (gain) |
|---|---|---|
| MobileNetV2 w0.5 | 60.63 -> 66.88 (**+6.25**) | 74.58 -> 75.03 (+0.45) |
| MobileNetV3-Small | 78.20 -> 79.84 (+1.64) | 78.34 -> 80.02 (+1.68) |
| MobileNetV2 w1.0 | 80.86 -> 81.71 (+0.85) | 80.46 -> 81.53 (+1.07) |
| MobileNetV2 w1.4 | 82.36 -> 83.61 (+1.25) | 82.18 -> 84.05 (+1.87) |

With AdamW the gain rises with width (+0.45, +1.07, +1.87), so the width trend is not only an optimizer effect.
The large SGD gain on w0.5 comes from a badly under-trained SGD baseline (60.63 vs 74.58): KD helps most when the
student baseline is weak or under-optimized (consistent with the from-scratch result below). Single seed.

### From scratch vs pretrained (CIFAR-100, test top-1, 20 epochs, single seed for scratch)

| Student | Init | Baseline | KD (T=2, alpha=0.9) | Gain (pp) |
|---|---|---|---|---|
| MobileNetV2 w0.5 | pretrained (3 seeds) | 74.71 | 75.17 | +0.46 |
| MobileNetV2 w0.5 | from scratch | 47.44 | 49.94 | +2.50 |
| MobileNetV2 w1.0 | pretrained (3 seeds) | 80.82 | 81.63 | +0.81 |
| MobileNetV2 w1.0 | from scratch | 57.33 | 62.93 | +5.60 |

Without ImageNet weights KD helps several times more than with them. From-scratch students are not converged after
20 epochs, so part of this gain may be faster learning from the teacher's soft labels; scratch results are single-seed
and the optimizer differs between w0.5 (AdamW) and w1.0 (SGD).

### PlantVillage (block split, 20 epochs, same recipes and KD setting, single seed, test top-1)

| Model | Params (M) | Baseline | KD (T=2, alpha=0.9) | Gain (pp) |
|---|---|---|---|---|
| ResNet-50 (teacher) | 23.59 | 99.05 | — | — |
| MobileNetV2 w0.5 | 0.74 | 98.72 | 98.77 | +0.05 |
| MobileNetV2 w1.0 | 2.27 | 98.96 | 99.27 | +0.31 |

The dataset is saturated (~99%): with 10,886 test images, +0.05 pp is about 5 images and +0.31 pp about 34, so there is
no clear KD effect here. The w1.0 KD student even exceeds its teacher, which together with the approximate block split
(possible near-duplicate leakage) means PlantVillage cannot separate the models well.

## Repository layout

```
configs/    teacher.yaml, student.yaml, kd.yaml
src/        data.py, models.py, kd_loss.py, train.py, evaluate.py, utils.py
scripts/    queues (v2 + v1 history), plot_results.py, export.py, quantize_tflite.py, eval_tflite.py, benchmark.py
results/    results_v2.csv, best_recipes_v2.csv, best_config_v2.json, kd_gain_v2.csv, alpha_probe_v2.csv,
            summary_table_v2.csv, results.csv + quick_tests.csv (v1 history), figures/
notebooks/  optional
report/     report.pdf (to do)
```
