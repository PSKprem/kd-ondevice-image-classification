"""P25 — Knowledge Distillation loss (Hinton et al. 2015, arXiv:1503.02531).

    L = alpha * T^2 * KL(softmax(z_s / T) || softmax(z_t / T))
        + (1 - alpha) * CE(z_s, y)

The T^2 factor keeps the soft-target gradient magnitude comparable to the hard
CE term (see Sec. 2 of the paper).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """Combined distillation + cross-entropy loss.

    Args:
        temperature: softmax temperature T (>= 1).
        alpha: weight on the soft (distillation) term, in [0, 1].
        label_smoothing: smoothing applied to the hard-target CE term.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> dict[str, torch.Tensor]:
        T = self.T

        # Soft targets: KL divergence between temperature-softened distributions.
        # F.kl_div expects log-probabilities for the input (student).
        s_log_prob = F.log_softmax(student_logits / T, dim=1)
        t_prob = F.softmax(teacher_logits / T, dim=1)
        kd = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T * T)

        # Hard targets: standard cross-entropy on the (un-scaled) student logits.
        ce = self.ce(student_logits, targets)

        loss = self.alpha * kd + (1.0 - self.alpha) * ce
        return {"loss": loss, "kd": kd.detach(), "ce": ce.detach()}
