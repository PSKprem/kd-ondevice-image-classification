# P25 — Knowledge Distillation for On-Device Image Classification

## Project Overview

| Field | Details |
|---|---|
| Project ID | P25 |
| Project Title | Knowledge Distillation for On-Device Image Classification |
| Section | Section B |
| Assigned Team | Team 25 |
| Domain / Type | Image |

## Team Members

| S. No. | Name | Email |
|---|---|---|
| 1 | Shivang Dubey | su-24137@sitare.org |
| 2 | Moirangthem Joel Singh | su-24080@sitare.org |
| 3 | Rajiv Kumar | su-24109@sitare.org |
| 4 | Prem Shankar Kushawaha | su-24100@sitare.org |

## Problem Statement

Compress a large teacher network into a tiny student network that runs on a phone with minimal accuracy loss.

## Suggested Datasets

- CIFAR-100
- PlantVillage

## Suggested Approach (Baseline → Advanced)

1. Train a ResNet-50 teacher network.
2. Train a MobileNetV2 student network normally (baseline, no KD).
3. Train the MobileNetV2 student with Knowledge Distillation (KD) and compare against the baseline.
4. Vary hyperparameters: Temperature ($T$) and Alpha ($\alpha$).
5. Measure latency, model size, and classification accuracy on-device.

## Research Angle (Paper-Worthy Contribution)

Analyze the accuracy-vs-model-size Pareto curve to answer: **When does Knowledge Distillation help the most?**

## Deliverables

1. Project Report
2. Project on GitHub (1 repo per group)

**Submission Deadline:** 20 Nov 2026, 11:59 PM IST

## Reference Papers

1. Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015 — https://arxiv.org/abs/1503.02531
2. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018 — https://arxiv.org/abs/1801.04381
