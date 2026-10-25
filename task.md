# P25 — Task Plan: Knowledge Distillation for On-Device Image Classification

> Brief: `project.md` · How to run: `README.md` · Last updated: 25 Oct 2026
> **Working alone** (the 4-member team split of the original plan is not used; say this honestly in the report).

## 1. Summary

| Field | Details |
|---|---|
| Project ID / Team | P25 / Section B, Team 25 |
| Goal | Compress a ResNet-50 teacher into a small MobileNet student that runs on a phone with minimal accuracy loss |
| Research question | **When does Knowledge Distillation help the most?** (accuracy-vs-model-size Pareto curve) |
| Datasets | CIFAR-100 (main), PlantVillage (second, smaller study) |
| Deliverables | Project Report + GitHub repo (1 per group) |
| **Deadline** | **20 Nov 2026, 11:59 PM IST** |
| Personal target | Finish by **10 Nov** (≈10 days buffer) |
| References | Hinton et al. 2015 (arXiv:1503.02531), Sandler et al. CVPR 2018 (arXiv:1801.04381) |

## 2. Decisions that are FINAL (do not re-debate)

| Topic | Decision |
|---|---|
| Approach | Transfer learning (ImageNet-pretrained), 128 px, 20 epochs, AMP. From-scratch only as a small side comparison |
| Protocol (v2) | CIFAR-100 train 50k → **45k train / 5k val** (fixed `split_seed 42`). Best epoch + recipe chosen on **val**; the 10k **test** set is only reported (`test_top1` = headline) |
| v1 results (`results/results.csv`) | Picked best epoch on test → optimistic. **History only; never use for headline numbers** |
| Teacher | ResNet-50, SGD lr 0.01, label smoothing 0 |
| Students | MobileNetV2 width 0.5 / 1.0 / 1.4 and MobileNetV3-Small (extra point for the Pareto curve). All pretrained |
| Recipe per student (chosen on val) | w0.5 → AdamW 1e-3 · w1.0 → SGD 0.01 · w1.4 → SGD 0.01 · MBV3-S → AdamW 1e-3 |
| KD setting | **One common setting T = 2, α = 0.9 for all students** (see reason below) |
| Fairness | Baseline vs KD: same seed, augmentation, epochs, recipe, starting weights |

Why T = 2, α = 0.9 for all: on the w1.0 grid its val score is 81.52 vs best 81.86 (difference is within noise; val SE ≈ 0.55 pp);
it does no harm on w0.5 (AdamW); it gave +1.7 … +2.0 pp on w1.4 / MBV3-S in v1.
The full 12-cell grid (T ∈ {1,2,4,8} × α ∈ {0.1,0.5,0.9}) on w1.0 is reported as the T × α heatmap; the differences between neighbouring cells are near seed noise, so claims must stay modest.

## 3. Results so far (CIFAR-100, v2, `results/results_v2.csv`)

| Model | Params (M) | Recipe | val top-1 | **test top-1** |
|---|---|---|---|---|
| ResNet-50 teacher | 23.71 | SGD 0.01 | 86.06 | **85.46** |
| MBV2 w0.5 (baseline) | 0.82 | AdamW | 74.90 | **74.58** |
| MBV2 w1.0 (baseline) | 2.35 | SGD | 80.74 | **80.86** |
| MBV2 w1.4 (baseline) | 4.50 | SGD | 82.58 | **82.36** |
| MBV3-Small (baseline) | 1.62 | AdamW | 78.32 | **78.34** |
| MBV2 w1.0 + KD (T2, α0.9) | 2.35 | SGD | 81.52 | **81.71** (+0.85) |

KD (T2, α0.9) results: w0.5 test 75.03 (+0.45), MBV3-S 80.02 (+1.68), w1.0 81.71 (+0.85), w1.4 83.61 (+1.25) — single seed, `results/kd_gain_v2.csv`. Gain is small and may be seed noise (w0.5 val gain is only +0.10; w1.0 +0.78 is borderline vs val noise ~0.55; w1.4 and MBV3-S gain >1 pp on both val and test are probably real). Within MobileNetV2 the gain rises with width (+0.45 -> +0.85 -> +1.25), but this is **confounded with the optimizer** (w0.5 and MBV3-S use AdamW, w1.0 and w1.4 use SGD, each chosen on val) and MBV3-S is a different architecture, so "gain depends only on size" must NOT be claimed. Partial de-confounding: in v1 all four students used AdamW and showed the same pattern (w0.5 ~0, w1.0 +1.4, w1.4 +2.0, MBV3-S +1.7; test-selected protocol, so optimistic). Put the optimizer/architecture confound in the report's Limitations. Single seed -> confirm with extra seeds (`--seeds`) before claiming it.
Preliminary v1 trend (test-selected, optimistic): KD gain ≈ 0 on w0.5 (3 seeds), ≈ +1.4 on w1.0, ≈ +2.0 on w1.4, ≈ +1.7 on MBV3-S → gain seems to grow with student size. Must be re-confirmed on v2.

## 4. Status checklist

### Phase 0 — Setup & code ✅
- [x] venv (Python 3.12.5), torch 2.5.1+cu121 (CUDA OK on RTX 4050), timm, pandas, matplotlib, pyyaml, thop; `huggingface_hub` pinned 0.26.2
- [x] `src/` (data, models, kd_loss, train, evaluate, utils) and `scripts/` written, smoke test passed
- [x] CIFAR-100 in `D:\kd-project\data`
- [x] Git: v2 code + results committed (history: `5a8b8f4`, `93796a2`, `2fedcba`)

### Phase 1 — CIFAR-100 v2: teacher, baselines, T × α sweep ✅
- [x] Teacher ResNet-50 (v2)
- [x] 8 baselines (4 students × SGD / AdamW) → `results/best_recipes_v2.csv`
- [x] Full T × α grid on w1.0 (12 runs) → `results/best_config_v2.json`, `results/figures/heatmap_v2.png`
- [x] v1 experiments (history only): sweep on w0.5, recipe / label-smoothing / seed checks, from-scratch w0.35 / 0.5 / 0.75

### Phase 2 — KD on every student (Pareto data) ✅
- [x] `scripts\queue_v2_exp4.ps1` finished 25 Oct (2.1 h, 7 runs, no failures): KD (T2, α0.9) on w0.5 / w1.4 / MBV3-S + alpha probe (α 0.1, 0.5 at T2 on w0.5 and w1.4; discussion only). Files: `results/kd_gain_v2.csv`, `results/alpha_probe_v2.csv`
- Alpha probe (test gain vs baseline, pp, single seed; α = 0.1 / 0.5 / 0.9): w0.5 +0.32 / +0.40 / +0.45 · w1.0 +0.52 / +1.06 / +0.85 · w1.4 +0.28 / +1.53 / +1.25. α = 0.5 is slightly better than the common α = 0.9 on w1.0 and w1.4 (val agrees: w1.4 +1.04 vs +1.00), but the 0.2–0.3 pp difference is within noise; α = 0.1 is clearly weakest on the larger students; w0.5 is flat and small. Decision stays: common T2 / α0.9 (mention as a limitation)
- [x] Seed check done 25 Oct (`scripts\queue_v2_seeds.ps1`, 8 runs, 1.9 h): seeds 42 / 0 / 1 on w0.5 and w1.0. Baseline vs KD test (mean ± std): w0.5 74.71 ± 0.16 -> 75.17 ± 0.15 (gain +0.46; per seed +0.45 / +0.44 / +0.49); w1.0 80.82 ± 0.07 -> 81.63 ± 0.18 (gain +0.81; per seed +0.85 / +0.89 / +0.68). Gains are small but positive for every seed and larger than seed spread (~0.1–0.2 pp) -> likely real, for these two students. This replaces the earlier v1 impression that w0.5 gains ≈ 0. w1.4 and MBV3-S remain single seed (optional: `--seeds --seed-students w1.4 mbv3s`, ~1.3 h)
- [x] Optimizer de-confound done 25 Oct (`scripts\queue_recipe_swap.ps1`, 4 KD runs, 1.1 h, single seed 42; `results/recipe_swap_v2.csv`). KD gain on test (pp), SGD / AdamW: w0.5 **+6.25** / +0.45 · w1.0 +0.85 / +1.07 · w1.4 +1.25 / +1.87 · MBV3-S +1.64 / +1.68. Findings: (1) with AdamW the gain rises monotonically with width (+0.45 -> +1.07 -> +1.87), so the width trend is not just an optimizer artefact; (2) with SGD it holds for w1.0 -> w1.4 but w0.5 is a huge outlier (+6.25) because its SGD baseline is badly under-trained (60.63 vs 74.58 with AdamW) — same lesson as from-scratch: KD helps a lot when the student baseline is weak/under-optimized; (3) MBV3-S gain is about the same under both optimizers (+1.6). Caveat: these 4 new runs are single seed

### Phase 3 — Plots & tables
- [x] `scripts/plot_results.py` rewritten for v2 and run on the final data -> `pareto_v2.png`, `kd_gain_v2.png`, `heatmap_v2.png`, `alpha_probe_v2.png`, `results/summary_table_v2.csv`
- [x] Figures: Pareto (test acc vs params, baseline vs KD, + teacher), KD gain vs width, T × α heatmap, alpha probe, summary table
- [x] From-scratch vs pretrained redone on v2 (25 Oct, `scripts\queue_scratch.ps1`, 4 runs, 0.9 h; 20 epochs, same recipes/teacher/KD T2 α0.9, single seed; results `results/results_scratch_v2.csv`, `results/summary_scratch_v2.csv`). Test top-1 baseline -> KD: w0.5 47.44 -> 49.94 (**+2.50**, val +3.64) · w1.0 57.33 -> 62.93 (**+5.60**, val +5.56). Pretrained gains were +0.46 / +0.81, so **KD helps ~5-7x more without pretrained weights** — a strong, clear result (effect size far above seed spread ~0.2 pp). Caveats: single seed; 20 epochs is not converged for from-scratch (part of the KD gain may be faster learning from soft labels); w0.5 uses AdamW and w1.0 SGD (optimizer confound again)

### Phase 4 — On-device measurement (core of the problem statement) ⏳ do EARLY
- [x] Separate export env `D:\kd-project\.venv-export` (CPU torch, TF 2.21, onnx2tf 2.6.9, ai-edge-quantizer); `requirements-export.txt` now holds the VERIFIED versions and what failed. Training venv untouched. All downloads went to D: (pip cache `D:\kd-project\cache\pip`, temp `D:\kd-project\tmp`)
- [x] Export test on the w1.0 KD checkpoint: PyTorch -> ONNX -> TFLite FP32 (8.95 MiB) -> INT8 dynamic (2.5 MiB) / INT8 static (2.7 MiB). CIFAR-100 test images: FP32 81.10% (n=1000, 100% agreement with PyTorch), INT8 dynamic 83.5% (n=200, 96.5% agreement), INT8 static 83.0% (n=500, 94.8% agreement). Small n -> run the full 10k test set for the final table
- Pipeline: `export.py --onnx-only` -> `onnx2tf -i model.onnx -o dir -b 1` -> `quantize_tflite.py` -> `eval_tflite.py`. Test artifacts in `D:\kd-project\export\test_w1.0\`
- [ ] Export 4 student architectures (KD checkpoints; baseline and KD have identical size/latency, only accuracy differs; check INT8 accuracy for both) + teacher FP32 for reference
- [ ] Per model record: file size (MB), params, FLOPs, median latency (ms, ~100 runs), accuracy of exported model
- [x] Phone: Android phone available, USB debugging ON; platform-tools installed at `D:\Downloads\platform-tools-latest-windows\platform-tools\adb.exe` (not on PATH), phone connection checked by user (25 Oct)
- [ ] Still to do (user will say when): push TFLite `benchmark_model` binary to the phone, run via `scripts/benchmark.py` (pass the adb path or add it to PATH)
- [ ] Record phone model, Android version, CPU, threads used and whether XNNPACK / GPU delegate is on (needed for the report's latency table)
- [ ] Laptop CPU latency may be added as an extra column, clearly labelled as a proxy (not the main on-device number)

### Phase 5 — PlantVillage (smaller study) ✅ (runs done; optional seeds)
- [x] Kaggle account + legacy API key (`C:\Users\Victus\.kaggle\kaggle.json`, never commit); kaggle CLI 2.2.5 in separate venv `D:\kd-project\.venv-kaggle`
- [x] Downloaded `abdallahalidev/plantvillage-dataset` (2.04 GB zip) and extracted ONLY `color/` to `D:\kd-project\data\plantvillage_raw\plantvillage dataset\color`: 38 classes, 54,305 images (152 to 5,507 per class). Config still points to `D:/kd-project/data/plantvillage` -> set `plantvillage_dir` to the `color` folder
- Finding: filenames are `<uuid>___<code>.JPG` (or `<code>.JPG`); the code is a per-image id (53,091 unique codes), so **leaf identity is not in the filename** -> exact per-leaf grouping is impossible. Plan: block split (consecutive ids within a class go to the same split); say in the report that it is approximate
- [x] `_build_plantvillage` implemented in `src/data.py` (block split: per class, blocks of 20 consecutive ids go whole to train / val / test = 37,994 / 5,425 / 10,886 (70/10/20), 38 classes, no overlap, deterministic, `split_seed` 42). Extra flags `--dataset plantvillage --plantvillage-dir ...`; configs now point to the `color` folder. Smoke test passed end-to-end (38-class head). Check: adjacent ids are more similar than random same-class images (pixel diff 32 vs 39), so blocks reduce but cannot fully remove leaf leakage -> state as a limitation
- [x] Runs (5) finished 25 Oct (`scripts\queue_pv.ps1`, 1.4 h, no failures; 20 epochs, same recipes as CIFAR-100, recipe not re-tuned): results in `results/results_pv_v2.csv`, summary `results/summary_pv_v2.csv`. Test top-1 (single seed): teacher ResNet-50 99.05 (val 98.84) · w0.5 baseline 98.72 -> KD 98.77 (+0.05; val 98.19 -> 98.43) · w1.0 baseline 98.96 -> KD 99.27 (+0.31; val 98.88 -> 99.08)
- Interpretation (be honest in the report): the dataset is saturated (~99%). The test set has 10,886 images, so +0.05 is ~5 images (noise) and +0.31 is ~34 images (borderline, single seed). The w1.0 KD student (99.27) even beats its teacher (99.05), which together with the approximate block split (near-duplicate leakage, see above) suggests PlantVillage cannot separate the models well -> do NOT claim KD helps here; say "no clear effect"

### Phase 6 — Optional (only if time remains)
- [ ] Low-data study (10% / 25% / 100% training data) on one student
- [ ] Extra seeds for more students

### Phase 7 — Report & GitHub
- [ ] Report draft (outline in section 9), final PDF in `report/report.pdf`
- [ ] README final check (commands work from a clean clone; results tables match CSVs)
- [ ] Push to GitHub (no `*.pt`, no secrets); keep commit history meaningful
- [ ] Upload best checkpoints + `.tflite` to Hugging Face / GitHub Release; backup to OneDrive / Drive
- [ ] Update this file when each phase is finished

## 5. Timeline (today = 25 Oct, deadline = 20 Nov)

| Dates | Plan |
|---|---|
| 25 Oct | Phase 2 (start queue, ~50 min) + Phase 3 plot script fix while it runs |
| 26–28 Oct | Phase 4: install adb + check phone connection, export env + single-model export test, first latency on the phone |
| 29 Oct – 1 Nov | Phase 5: PlantVillage download, grouped split, overnight runs |
| 2–4 Nov | Phase 4: export all models, benchmark; Phase 3 final figures |
| 5–8 Nov | Report draft |
| 9–10 Nov | Report final, GitHub push, backups (**personal target**) |
| 11–20 Nov | Buffer only (re-runs, fixes). Submit before 20 Nov |

If a phase slips: Phase 6 is dropped first, then the from-scratch redo, then PlantVillage seeds. Never drop Phase 4 or Phase 7.

## 6. Commands

```powershell
# Phase 2 (resumable: run again after any interruption)
powershell -ExecutionPolicy Bypass -File scripts\queue_v2_exp4.ps1
# Phase 2 optional seeds
.\.venv\Scripts\python.exe -u scripts/queue_v2_exp4.py --seeds
# Dry run (prints commands, trains nothing)
.\.venv\Scripts\python.exe -u scripts/queue_v2_exp4.py --dry-run
```

Logs: `D:\kd-project\logs\`. Checkpoints: `D:\kd-project\runs_v2\<run_name>\best.pt`, `last.pt`.

## 7. Hardware, storage, security

- RTX 4050 6 GB (use AMP), i7-13620H, 16 GB RAM; data + checkpoints on D:. Laptop plugged in, sleep off, cooling pad, Windows Update paused while training.
- Git holds code, configs, `results/*.csv`, figures, report (<60 MB). Checkpoints, datasets, `.onnx`, `.tflite` stay out (`.gitignore`).
- Never commit Kaggle key or Hugging Face token (`.env`, `kaggle.json`, `*token*.txt` are ignored).
- Stop anytime with `Ctrl+C`; restart with `--resume` / re-run the queue — loses at most 1 epoch.

## 8. Risks

| Risk | Mitigation |
|---|---|
| TFLite export fails on Windows | Test early (Phase 4); fall back to `onnx2tf` / `ai-edge-torch` / Colab / WSL |
| `benchmark_model` binary / adb setup problems | Install platform-tools early; test with any small .tflite first |
| PlantVillage leakage | No leaf id available -> block split of consecutive ids (approximate); state as a limitation |
| KD gains within noise (≈ 0.3–1 pp) | Report full grid, add seeds where possible, keep claims modest |
| Laptop sleep / thermal throttling / updates | Settings in section 7, resumable queues |
| GitHub 100 MB limit | Checkpoints never in git |

## 9. Report outline

1. Introduction · 2. Related work (Hinton 2015, MobileNetV2) · 3. Method (KD loss, T, α)
4. Experimental setup (datasets, v2 train/val/test protocol, models, recipes, hardware)
5. Results: teacher vs baseline vs KD (accuracy, size, latency); T × α heatmap; accuracy-vs-size Pareto (baseline vs KD); KD gain vs width; pretrained vs from-scratch; CIFAR-100 vs PlantVillage; on-device table
6. Discussion — *when does KD help the most?* (including honest limits: small gains, seed noise, single common T/α, v1→v2 protocol change)
7. Conclusion · 8. References

## 10. Repository layout (actual)

```
configs/   teacher.yaml, student.yaml, kd.yaml
src/       data.py, models.py, kd_loss.py, train.py, evaluate.py, utils.py
scripts/   train/queue: queue_v2_night1.py|.ps1, queue_v2_exp4.py|.ps1   (v2, current)
           history  : queue_night3, queue_exp4, queue_seeds, sweep*, overnight_night1, pretrained_baselines (v1)
           tools    : plot_results.py, export.py, benchmark.py
results/   results_v2.csv, best_recipes_v2.csv, best_config_v2.json, kd_gain_v2.csv (after Phase 2),
           results.csv + quick_tests.csv (v1 history), figures/
report/    report.pdf (to do)
notebooks/ optional
```
