# P25 — Knowledge Distillation for On-Device Image Classification

> Source of truth: the course project sheet (row P25). This file mirrors that row; if they ever differ, the sheet wins.

## Project Overview

| Field | Details |
|---|---|
| Project ID | P25 |
| Assigned Team | 25 |
| Section | section-b (Section B) |
| Project Title | Knowledge Distillation for On-Device Image Classification |
| Type | Image |

## Team Members

| S. No. | Name | Email |
|---|---|---|
| 1 | Shivang Dubey | su-24137@sitare.org |
| 2 | Moirangthem Joel Singh | su-24080@sitare.org |
| 3 | Rajiv Kumar | su-24109@sitare.org |
| 4 | Prem Shankar Kushawaha | su-24100@sitare.org |

## Problem Statement

Compress a large teacher network into a tiny student that runs on a phone with minimal accuracy loss.

## Suggested Dataset

CIFAR-100 or PlantVillage

(One of the two is enough; this project uses both — CIFAR-100 as the main study and PlantVillage as a smaller second study.)

## Suggested Approach (baseline → advanced)

Train ResNet-50 teacher → MobileNetV2 student trained normally vs with KD (vary temperature, alpha) → measure latency/size.

## Research Angle (what makes it paper-worthy)

Accuracy-vs-model-size Pareto curve; when does KD help most?

## Deliverables

1. Project Report
2. Project on GitHub (1 repo per group)

**Submission Deadline:** 20 Nov 2026, 11:59 PM IST

*(Deliverables and deadline are not part of the sheet row above; they are kept from the earlier version of this file.)*

## Reference Paper(s)

[1] Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015 — https://arxiv.org/abs/1503.02531

[2] Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018 — https://arxiv.org/abs/1801.04381
