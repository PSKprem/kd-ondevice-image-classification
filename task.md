# P25 — Task Plan: Knowledge Distillation for On-Device Image Classification

> Project brief: see `project.md`

## 1. Project Summary

| Field | Details |
|---|---|
| Project ID | P25 |
| Title | Knowledge Distillation for On-Device Image Classification |
| Section / Team | Section B / Team 25 |
| Domain | Image |
| Members | Shivang Dubey (su-24137), Moirangthem Joel Singh (su-24080), Rajiv Kumar (su-24109), Prem Shankar Kushawaha (su-24100) |
| Goal | Compress a ResNet-50 teacher into a MobileNetV2 student that runs on a phone with minimal accuracy loss |
| Research question | When does Knowledge Distillation help the most? (Accuracy-vs-model-size Pareto curve) |
| Datasets | CIFAR-100, PlantVillage |
| References | Hinton et al. 2015 (arXiv:1503.02531), Sandler et al. CVPR 2018 (arXiv:1801.04381) |

## 2. Deliverables & Deadline

- [ ] Project Report
- [ ] Project on GitHub (1 repo per group)

**Deadline: 20 Nov 2026, 11:59 PM IST**
Target finish: 5–10 Nov 2026 (leaves ~10 days buffer). Working alone is allowed.

## 3. Hardware (checked)

| Component | Spec | Status |
|---|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB VRAM, CUDA 13.1 driver | OK — use AMP (mixed precision) |
| CPU | Intel i7-13620H (10 cores / 16 threads) | OK |
| RAM | 16 GB | OK |
| Disk | C: 165 GB free, D: 267 GB free | OK — store data/checkpoints on D: |
| Python | 3.14.5 + 3.12.5 installed | Project venv `.venv` uses Python 3.12.5 (no extra download needed) |

Training precautions:
- Laptop plugged in, sleep disabled, lid-close = "do nothing"
- Hard flat surface / cooling pad (avoid thermal throttling)
- Pause Windows Update during training week
- Avoid games / GPU-heavy apps while training (light work is fine)

## 4. Chosen Approach: Option C (Hybrid)

Main results with **transfer learning** (ImageNet-pretrained, ~128 px inputs, 10–30 epochs) + one small **from-scratch** experiment for comparison (adds to the research angle).

| Option | Total GPU time | Alone (full) | Team |
|---|---|---|---|
| A: From scratch | ~40–60 h | 3–4 weeks | ~2 weeks |
| B: Transfer learning | ~12–15 h | 2–2.5 weeks | 1–1.5 weeks |
| **C: Hybrid (chosen)** | **~20–25 h** | **2.5–3 weeks** | **~1.5 weeks** |
| C with AI assistance | ~20–25 h | **~7 night sessions** | — |

Note: torchvision has pretrained MobileNetV2 only at width 1.0; timm has some other widths — verify during setup.

## 5. Experiments

| # | Experiment | Runs | Est. time / run |
|---|---|---|---|
| 1 | ResNet-50 teacher (fine-tune) | 1 | 30–60 min |
| 2 | MobileNetV2 baseline, no KD (widths 0.35 / 0.5 / 0.75 / 1.0) | 4 | 15–40 min |
| 3 | T × α sweep on one width (T ∈ {1,2,4,8}, α ∈ {0.1,0.5,0.9}) | 12 | ~15 min |
| 4 | Final KD runs with best (T, α), all widths | 4 | ~40 min |
| 5 | PlantVillage: teacher + students (baseline & KD) | ~9 | 0.5–2 h |
| 6 | From-scratch comparison (small) | few | 1.5–2 h |
| 7 | Optional: low-data (10% / 25% / 100%) | ~6 | ~1 h |

KD loss (Hinton 2015):
`L = α · T² · KL(softmax(z_s/T) || softmax(z_t/T)) + (1 − α) · CE(z_s, y)`

Fairness rules: same seed, augmentation, epochs, and starting weights for baseline vs KD.

## 6. On-Device Measurement

- Export: PyTorch → ONNX → TFLite (FP32 + INT8). If Windows tools fail, do this step in Colab/WSL.
- Benchmark on Android phone via adb (`benchmark_model`), USB debugging ON.
- Record per model: file size (MB), params, FLOPs, median latency (ms, ~100 runs), accuracy of exported model.

## 7. Night-by-Night Plan (2–3 h sessions)

| Night | Your session | Overnight (unattended) | Done |
|---|---|---|---|
| 1 | Setup env, write code, 1-epoch test | Teacher + 4 baselines | [ ] |
| 2 | Check results, understand code | T × α sweep | [ ] |
| 3 | Pick best T/α, review plots | Final KD runs | [ ] |
| 4 | Prepare PlantVillage | PlantVillage + from-scratch | [ ] |
| 5 | Export TFLite, phone benchmark | — | [ ] |
| 6 | Plots, tables, report draft | — | [ ] |
| 7 | Review report, GitHub, buffer | — | [ ] |

Missed a night? Nothing is lost — resume the next night.

Night 1 progress (24 Oct 2026):
- [x] venv (Python 3.12.5), torch 2.5.1+cu121 (CUDA verified on RTX 4050), timm, pandas, matplotlib, tqdm, pyyaml, thop
- [x] All code in `src/` and `scripts/` written
- [x] CIFAR-100 downloaded to `D:\kd-project\data` (MD5 verified)
- [x] 1-epoch smoke test passed (GPU + AMP, checkpoints saved)
- [x] Overnight: teacher + 4 baselines (teacher 86.62, w1.0 81.76; w0.35/0.5/0.75 turned out from-scratch → relabelled `_scratch`)
- [x] Fix: huggingface_hub 2.3.0 broke timm pretrained download → pinned 0.26.2; models.py now errors instead of silently using random init
- [x] Pretrained baselines (24 Oct): w0.5 60.80, w1.4 82.89, MobileNetV3-Small 79.20 (`scripts/pretrained_baselines.ps1`)
- Student set for main results (all pretrained): MBV2 w0.5 / w1.0 / w1.4 + MobileNetV3-Small; scratch runs = from-scratch comparison
- Deferred to Night 5: onnx, onnxruntime, onnx-tf, tensorflow

## 8. Stop & Resume

- Checkpoint `last.pt` saved every epoch (model, optimizer, scheduler, AMP scaler, epoch, best accuracy).
- Stop anytime with `Ctrl+C`; restart with `--resume` → loses at most 1 epoch (~1–3 min).
- Sweep script skips runs already in `results/results.csv`.
- Internet is NOT needed during training (only for initial downloads).

## 9. Downloads Needed (one-time, ~5–7 GB)

- [x] Python 3.12 env (used existing 3.12.5 instead of installing 3.11)
- [x] PyTorch + torchvision, CUDA build (~2.5–3 GB, CUDA bundled)
- [x] timm
- [x] numpy, pandas, matplotlib, tqdm, pyyaml
- [x] thop or fvcore (params/FLOPs)
- [ ] onnx, onnxruntime, TFLite converter
- [ ] kaggle CLI (PlantVillage)
- [ ] huggingface_hub
- [ ] Android platform-tools (adb)
- [x] Git (2.47.0) + GitHub CLI
- [x] CIFAR-100 (~0.2 GB)  - [ ] PlantVillage (~1–2 GB)

Accounts (free): [ ] GitHub [ ] Kaggle (API key) [ ] Hugging Face (token)

## 10. Repository Structure (~15–20 files)

```
kd-on-device/
├── README.md
├── project.md
├── task.md
├── requirements.txt
├── .gitignore
├── configs/        teacher.yaml, student.yaml, kd.yaml
├── src/            data.py, models.py, kd_loss.py, train.py, evaluate.py, utils.py
├── scripts/        sweep.py, export.py, benchmark.py, plot_results.py
├── results/        results.csv, figures/
├── notebooks/      analysis.ipynb (optional)
└── report/         report.pdf
```

## 11. Storage Plan

| Item | Location | Size | Upload |
|---|---|---|---|
| Python env + PyTorch | Laptop | ~5 GB | No |
| Datasets | Laptop (D:\kd-project) | ~1.5–2 GB | No (link in README) |
| All checkpoints | Laptop (D:\kd-project) | ~2–3 GB | No |
| Code, configs, results.csv, plots, report | Laptop + GitHub | <60 MB | GitHub |
| Final best models + .tflite | Laptop + Hugging Face | ~200–300 MB | Hugging Face (or GitHub Release) |
| Backup of best checkpoints | Google Drive / OneDrive | ~300 MB | Recommended |

Security: never commit the Hugging Face token or Kaggle key; use `huggingface-cli login` / env variables; keep secrets in `.gitignore`.

## 12. Report Outline

1. Introduction
2. Related Work (Hinton 2015, MobileNetV2)
3. Method (KD loss, T and α)
4. Experimental Setup (datasets, models, hardware, training details)
5. Results
   - Teacher vs baseline student vs KD student (accuracy, size, latency)
   - T × α heatmap
   - Accuracy-vs-size Pareto curve (baseline vs KD)
   - KD gain vs width; pretrained vs from-scratch; CIFAR-100 vs PlantVillage
6. Discussion — When does KD help the most?
7. Conclusion
8. References

## 13. Optional Team Split

| Member | Responsibility |
|---|---|
| Shivang Dubey | Teacher training, shared training code |
| Moirangthem Joel Singh | Student baselines, width variants |
| Rajiv Kumar | KD implementation, T × α sweep |
| Prem Shankar Kushawaha | PlantVillage, export, phone benchmarking |
| All | Analysis and report |

Each member should commit their own part to GitHub (graders may check commit history) and be able to explain the code and results.

## 14. Risks

- Laptop sleep / Windows Update restarts → disable sleep, pause updates, use `--resume`
- Thermal throttling → cooling, hard surface
- TFLite export version issues → budget extra ½ day; fallback to Colab/WSL
- Missing pretrained weights for some MobileNetV2 widths → train from scratch or pick available widths
- GitHub 100 MB file limit → keep checkpoints out of git
