# P25 — Task Plan: Knowledge Distillation for On-Device Image Classification

> Brief: `project.md` (mirrors the official project sheet) · How to run: `README.md` · Report: `report/report.pdf` · Last updated: 19 Nov 2026
> **Working alone** (the 4-member team split of the original plan is not used; the report says this honestly).

## 0. Current status (read this first)

**All experiments, the export, the phone benchmark, the report (`report/report.md` + `report/report.pdf`) and the GitHub push are DONE.**
Every requirement of the project sheet is met (see report Section 2.4). What is left is review and delivery only:

| # | Remaining task | Who | Needed? |
|---|---|---|---|
| 1 | Proof-read `report/report.pdf` | user | **Yes** |
| 2 | Decide how the grader gets access to the **private** GitHub repo (make public or invite) | user | **Yes** |
| 3 | Add roll numbers / course info to the report if the submission form asks for them | user (tell the agent) | Only if asked |
| 4 | ~~Back up checkpoints and `.tflite` files to Google Drive~~ **Done** (link in README). User: check in Drive that the folder size is ≈ 3.8 GB | user | Done |
| 5 | Phone: Developer options -> "Revoke USB debugging authorizations" | user | Recommended (security) |
| 6 | README check from a clean clone (do the commands work?) | agent, ~15 min | Optional |
| 7 | FLOPs per model; laptop CPU latency as a proxy column; seeds for the single-seed side experiments | agent | Optional (not required by the sheet) |

## 1. Summary

| Field | Details |
|---|---|
| Project ID / Team | P25 / Section B, Team 25 |
| Goal | Compress a ResNet-50 teacher into a small MobileNet student that runs on a phone with minimal accuracy loss |
| Research question | **When does Knowledge Distillation help the most?** (accuracy-vs-model-size Pareto curve) |
| Datasets | CIFAR-100 (main), PlantVillage (second, smaller study). The sheet says "CIFAR-100 **or** PlantVillage"; both were done |
| Deliverables | Project Report + GitHub repo (1 per group) — both exist |
| **Deadline** | **20 Nov 2026, 11:59 PM IST** |
| Personal target | Finish by **10 Nov** |
| References | Hinton et al. 2015 (arXiv:1503.02531), Sandler et al. CVPR 2018 (arXiv:1801.04381) |

## 2. Decisions that are FINAL (do not re-debate)

| Topic | Decision |
|---|---|
| Approach | Transfer learning (ImageNet-pretrained), 128 px, 20 epochs, AMP. From-scratch only as a side comparison |
| Protocol (v2) | CIFAR-100 train 50k → **45k train / 5k val** (fixed `split_seed 42`). Best epoch + recipe chosen on **val**; the 10k **test** set is only reported (`test_top1` = headline) |
| v1 results (`results/results.csv`) | Picked best epoch on test → optimistic. **History only; never use for headline numbers** |
| Teacher | ResNet-50, SGD lr 0.01, label smoothing 0 |
| Students | MobileNetV2 width 0.5 / 1.0 / 1.4 and MobileNetV3-Small. All pretrained |
| Recipe per student (chosen on val) | w0.5 → AdamW 1e-3 · w1.0 → SGD 0.01 · w1.4 → SGD 0.01 · MBV3-S → AdamW 1e-3 |
| KD setting | **One common setting T = 2, α = 0.9 for all students** |
| Fairness | Baseline vs KD: same seed, augmentation, epochs, recipe, starting weights |
| Final INT8 recipe | **Mixed**: depthwise convs float32, other convs + classifier int8 (plain dynamic INT8 breaks w0.5 / MBV3-S) |

Why T = 2, α = 0.9 for all: on the w1.0 grid its val score is 81.52 vs best 81.86 (within noise; val SE ≈ 0.55 pp);
high-T / high-α settings hurt the small w0.5 in v1. The full 12-cell grid is reported as the T × α heatmap; neighbouring cells differ by noise only.

## 3. Final results (details and caveats in the report)

**CIFAR-100, test top-1, mean ± std over 3 seeds (42 / 0 / 1), KD T = 2, α = 0.9** (`results/kd_gain_v2.csv`)

| Model | Params (M) | Baseline | KD | Gain (pp) |
|---|---|---|---|---|
| ResNet-50 teacher (1 seed) | 23.71 | 85.46 | — | — |
| MBV2 w0.5 (AdamW) | 0.82 | 74.71 ± 0.16 | 75.17 ± 0.15 | +0.46 |
| MBV3-Small (AdamW) | 1.62 | 78.24 ± 0.26 | 79.77 ± 0.50 | +1.53 |
| MBV2 w1.0 (SGD) | 2.35 | 80.82 ± 0.07 | 81.63 ± 0.18 | +0.81 |
| MBV2 w1.4 (SGD) | 4.50 | 82.36 ± 0.02 | 84.06 ± 0.40 | +1.70 |

Gain is positive for every student and every seed. Side experiments (single seed): from scratch +2.50 / +5.60 pp (w0.5 / w1.0);
optimizer swap shows the width trend also with AdamW only (+0.45 / +1.07 / +1.87); low-data mixed (w1.0 +2.03 at 25 %, w0.5 −0.22 at 10 %);
PlantVillage saturated (~99 %), no clear effect (+0.05 / +0.31).

**On the phone** (Samsung Galaxy F22, 1 thread, median of 3): distilled w1.0 INT8-mixed = 2.62 MiB, 18.0 ms, 81.78 % vs teacher FP32 = 90.39 MiB, 225.4 ms, 85.46 %
→ **34× smaller, 12.5× faster, −3.7 pp**.

## 4. Status checklist

### Phase 0 — Setup & code ✅
- [x] venv (Python 3.12.5), torch 2.5.1+cu121 (CUDA OK on RTX 4050), timm, pandas, matplotlib, pyyaml, thop; `huggingface_hub` pinned 0.26.2
- [x] `src/` (data, models, kd_loss, train, evaluate, utils) and `scripts/` written, smoke test passed
- [x] CIFAR-100 in `D:\kd-project\data`

### Phase 1 — CIFAR-100 v2: teacher, baselines, T × α sweep ✅
- [x] Teacher ResNet-50 (v2): val 86.06 / test 85.46
- [x] 8 baselines (4 students × SGD / AdamW) → `results/best_recipes_v2.csv`
- [x] Full T × α grid on w1.0 (12 runs) → `results/best_config_v2.json`, `results/figures/heatmap_v2.png`
- [x] v1 experiments (history only): sweep on w0.5, recipe / label-smoothing / seed checks, from-scratch w0.35 / 0.5 / 0.75

### Phase 2 — KD on every student (Pareto data) ✅
- [x] KD (T2, α0.9) on all four students + alpha probe (α 0.1 / 0.5 at T2 on w0.5 and w1.4) — `scripts\queue_v2_exp4.ps1`, `results/kd_gain_v2.csv`, `results/alpha_probe_v2.csv`
- Alpha probe (test gain, pp, single seed; α = 0.1 / 0.5 / 0.9): w0.5 +0.32 / +0.40 / +0.45 · w1.0 +0.52 / +1.06 / +0.85 · w1.4 +0.28 / +1.53 / +1.25. α = 0.5 slightly better than 0.9 for the larger students but within noise; decision stays T2 / α0.9 (limitation in the report)
- [x] 3 seeds (42 / 0 / 1) for **all four** students — `scripts\queue_v2_seeds.ps1` (default w0.5 w1.0; `-Students "w1.4 mbv3s"`). Per-seed paired gains: w0.5 +0.45 / +0.44 / +0.49 · w1.0 +0.85 / +0.89 / +0.68 · w1.4 +1.25 / +2.02 / +1.84 · MBV3-S +1.68 / +2.15 / +0.76
- [x] Optimizer de-confound — `scripts\queue_recipe_swap.ps1`, `results/recipe_swap_v2.csv` (single seed). KD gain, SGD / AdamW: w0.5 **+6.25** / +0.45 · w1.0 +0.85 / +1.07 · w1.4 +1.25 / +1.87 · MBV3-S +1.64 / +1.68. Width trend holds with AdamW alone; the w0.5 SGD outlier comes from an under-trained SGD baseline (60.63 vs 74.58)

### Phase 3 — Plots & tables ✅
- [x] `scripts/plot_results.py` (v2) → `pareto_v2.png`, `kd_gain_v2.png`, `heatmap_v2.png`, `alpha_probe_v2.png`, `results/summary_table_v2.csv`
- [x] `scripts/plot_report_figures.py` → `fig_latency_pareto.png`, `fig_size_pareto.png`, `fig_lowdata.png`, `fig_scratch.png`, `fig_optimizer.png`
- [x] From-scratch vs pretrained (v2) — `scripts\queue_scratch.ps1`, `results/summary_scratch_v2.csv` (single seed). w0.5 47.44 → 49.94 (**+2.50**) · w1.0 57.33 → 62.93 (**+5.60**). Caveat: 20 epochs not converged for scratch

### Phase 4 — Export & on-device measurement ✅
- [x] Separate export env `D:\kd-project\.venv-export` (CPU torch, TF 2.21, onnx2tf 2.6.9, ai-edge-quantizer); verified versions + what failed in `requirements-export.txt`
- [x] Pipeline: `export.py --onnx-only` → `onnx2tf -b 1` (FP32 .tflite) → `quantize_tflite.py` (INT8) → `eval_tflite.py` (accuracy); all models via `scripts/export_all.py` → `results/export_eval_v2.csv`
- [x] FP32 TFLite == PyTorch predictions (100 % agreement on every model)
- [x] INT8 diagnosis (`scripts/diag_quant.py`): conversion is correct; the w0.5 loss comes from its depthwise layers (−2.60 pp when only depthwise is int8); MBV3-S breaks only with run-time activation quantization. Cause of the depthwise sensitivity NOT established
- [x] Fix = mixed INT8 (depthwise float32). Full 10k test set, KD / baseline: w0.5 74.67 / 74.43 · MBV3-S 79.52 / 78.04 · w1.0 81.78 / 80.48 · w1.4 83.55 / 82.26
- [x] Phone benchmark — `scripts/phone_benchmark.py`, `results/phone_latency_v2.csv`: Samsung Galaxy F22 (SM-E225F, Android 13, MediaTek Helio G80, 8 cores), CPU + XNNPACK, TFLite `benchmark_model` nightly (7 Feb 2025, SHA256 `5D45116E...FABA1374`), 13 models × 1 / 4 threads × 3 reps, 100 timed runs each, temperature 32.8–33.1 °C
- [x] Phone metadata (model, Android, CPU, threads, XNNPACK) recorded in the report (Sections 5.1, 7.5) and README
- [x] Per-model numbers: parameters (report Section 4.2), file sizes + exported accuracy (Section 7.3), latency (Section 7.5), headline trade-off (Section 7.6)
- [x] Phone clean-up: `/data/local/tmp/p25` deleted (verified), USB debugging no longer visible to adb. User still to tap "Revoke USB debugging authorizations"
- [ ] *Optional:* FLOPs per model (not computed; the sheet asks for latency/size only)
- [ ] *Optional:* laptop CPU latency as a clearly-labelled proxy column

### Phase 5 — PlantVillage (smaller study) ✅
- [x] Kaggle API key in `C:\Users\Victus\.kaggle\kaggle.json` (never commit); kaggle CLI in `D:\kd-project\.venv-kaggle`
- [x] `abdallahalidev/plantvillage-dataset`, only `color/` extracted: 38 classes, 54,305 images; configs point to `D:/kd-project/data/plantvillage_raw/plantvillage dataset/color`
- [x] Block split in `src/data.py` (no leaf id in the file names): blocks of 20 consecutive ids per class → 37,994 / 5,425 / 10,886 (approximate; limitation in the report)
- [x] 5 runs — `scripts\queue_pv.ps1`, `results/summary_pv_v2.csv` (single seed): teacher 99.05 · w0.5 98.72 → 98.77 (+0.05) · w1.0 98.96 → 99.27 (+0.31). Saturated → "no clear effect"
- [ ] *Optional:* extra seeds for PlantVillage

### Phase 6 — Optional extras ✅
- [x] Low-data study — `scripts\queue_lowdata.ps1`, `results/summary_lowdata_v2.csv` (single seed). Gain at 10 % / 25 % / 100 % data: w1.0 +1.12 / +2.03 / +0.85 · w0.5 −0.22 / +0.10 / +0.45 (mixed result)
- [x] 3 seeds for all four students of the main study (see Phase 2)
- [ ] *Optional:* extra seeds for the single-seed side experiments (optimizer swap, from scratch, low-data, alpha probe)

### Phase 7 — Report & GitHub
- [x] Report: `report/report.md` (12 sections + 3 appendices, numbers cross-checked against the CSVs), `report/report.pdf` (21 pages, `scripts/build_report_pdf.py`), "Section" wording, compliance table with the project sheet (Section 2.4)
- [x] Report revision 26 Oct (after an external review): full per-seed table for all four students in Appendix A.1; Section 7.6 breakdown of the 34× / 12.5× ratios (architecture 10.1× vs INT8 3.4×; INT8-vs-INT8 size 8.8×, teacher INT8 file 23.01 MiB, its accuracy/latency not measured); capacity-gap related work (Cho & Hariharan 2019, Mirzadeh et al. 2020) in Sections 3 and 8; from-scratch claims limited to "within the same 20-epoch budget". PDF rebuilt (21 pages)
- [x] `project.md` aligned with the official project sheet (row P25)
- [x] GitHub: everything committed and pushed to `PSKprem/kd-ondevice-image-classification` (private); no `*.pt`, `.tflite`, datasets or secrets in git
- [x] README updated with all results, export pipeline and phone benchmark
- [x] This file updated (19 Nov)
- [ ] **Proof-read the report** (user)
- [ ] **Grader access to the private repo** (user decision: make public or invite)
- [ ] Roll numbers / course info in the report, only if the submission form needs them
- [x] Hugging Face (26 Oct): final models uploaded to https://huggingface.co/Premshankark88/p25-kd-mobilenet-cifar100 (**private**): 4 distilled students + teacher, each `model.safetensors` (weights only, verified identical to `best.pt`), `config.json`, `model_fp32.tflite`, `model_int8_mixed.tflite` (students), plus `classes.json` and a model card. 263 MiB, 21 files; remote sizes verified, 2 files re-downloaded with matching SHA-256. Built by `scripts/prepare_hf_upload.py`, uploaded by `scripts/upload_hf.py` (`--public` to create public). Make it public together with the GitHub repo if the grader needs access
- [x] Full backup on Google Drive (26 Oct, by the user, account su-24100@sitare.org): folder "DeepLearnig(Project)" with `export/`, `logs/`, `runs_v2/` — https://drive.google.com/drive/folders/1x28bGIncOd_HTx_T96uGU1_t1ghtlG_t (shared "anyone with the link"; agent verified the three top-level folders are visible without login, but could not check the file count/size inside). Link added to README, report Section 11 and the HF model card
- [ ] *Optional:* README check from a clean clone

## 5. Timeline

| Dates | Plan | Actual |
|---|---|---|
| 24–26 Oct | Setup, all experiments, export, phone benchmark, report | ✅ done (well ahead of plan) |
| until 10 Nov | Proof-read, grader access, backups (**personal target**) | open |
| 11–20 Nov | Buffer only (fixes after proof-reading). Submit before 20 Nov | — |

## 6. Commands

```powershell
# training queues (all resumable: run again after any interruption; add --dry-run to the .py to only print commands)
powershell -ExecutionPolicy Bypass -File scripts\queue_v2_night1.ps1      # teacher, 8 baselines, T x alpha grid
powershell -ExecutionPolicy Bypass -File scripts\queue_v2_exp4.ps1        # KD on all students + alpha probe
powershell -ExecutionPolicy Bypass -File scripts\queue_v2_seeds.ps1 -Students "w1.4 mbv3s"
powershell -ExecutionPolicy Bypass -File scripts\queue_recipe_swap.ps1    # optimizer swap
powershell -ExecutionPolicy Bypass -File scripts\queue_scratch.ps1        # from scratch
powershell -ExecutionPolicy Bypass -File scripts\queue_lowdata.ps1        # low-data
powershell -ExecutionPolicy Bypass -File scripts\queue_pv.ps1             # PlantVillage

# figures and report
.\.venv\Scripts\python.exe scripts/plot_results.py
.\.venv\Scripts\python.exe scripts/plot_report_figures.py
D:\kd-project\.venv-export\Scripts\python.exe scripts/build_report_pdf.py   # report.md -> report.pdf

# export + phone (export venv / adb)
powershell -ExecutionPolicy Bypass -File scripts\export_all.ps1
.\.venv\Scripts\python.exe scripts/phone_benchmark.py --adb "D:/Downloads/platform-tools-latest-windows/platform-tools/adb.exe"
```

Logs: `D:\kd-project\logs\`. Checkpoints: `D:\kd-project\runs_v2\<run_name>\best.pt`, `last.pt`. Exports: `D:\kd-project\export\<run_name>\`.

## 7. Hardware, storage, security

- RTX 4050 6 GB (AMP), i7-13620H, 16 GB RAM; data, checkpoints, exports, caches and temp files on D:.
- Git holds code, configs, `results/*.csv`, figures, report (.md + .pdf). Checkpoints, datasets, `.onnx`, `.tflite`, `report/report.html` stay out (`.gitignore`).
- Never commit the Kaggle key or any token (`.env`, `kaggle.json`, `*token*.txt` are ignored).

## 8. Risks (status)

| Risk | Status |
|---|---|
| TFLite export fails on Windows | Solved: onnx2tf FP32 + ai-edge-quantizer |
| INT8 accuracy loss on small models | Solved: mixed INT8 (depthwise float32); cause not established |
| PlantVillage leakage | Approximate block split; stated as a limitation |
| KD gains within noise | 3 seeds on all students; gains positive in all 12 seed pairs |
| Checkpoints only on one disk | **Open** → backup (task 4 in Section 0) |
| Grader cannot open a private repo | **Open** → task 2 in Section 0 |

## 9. Repository layout (actual)

```
configs/   teacher.yaml, student.yaml, kd.yaml
src/       data.py, models.py, kd_loss.py, train.py, evaluate.py, utils.py
scripts/   queues (v2): queue_v2_night1, queue_v2_exp4, queue_v2_seeds, queue_recipe_swap, queue_scratch, queue_lowdata, queue_pv (.py/.ps1)
           export/phone: export.py, export_all.py|.ps1, quantize_tflite.py, eval_tflite.py, diag_quant.py, phone_benchmark.py, benchmark.py
           figures/report: plot_results.py, plot_report_figures.py, build_report_pdf.py
           history (v1): queue_night3, queue_exp4, queue_seeds, sweep*, overnight_night1, pretrained_baselines
results/   results_v2.csv, results_pv_v2.csv, results_scratch_v2.csv, results_lowdata_v2.csv, export_eval_v2.csv, phone_latency_v2.csv,
           kd_gain_v2.csv, summary_*_v2.csv, recipe_swap_v2.csv, alpha_probe_v2.csv, best_*_v2.*, results.csv + quick_tests.csv (v1), figures/
report/    report.md, report.pdf
```
