"""Custom losses for DWPose-style SimCC distillation."""

import torch
import torch.nn.functional as F
from torch import nn


class KLDistillationLoss(nn.Module):
    """KL divergence between student and teacher SimCC distributions."""

    def __init__(self, use_target_weight=True, loss_weight=1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, pred_simcc, teacher_simcc, target_weight=None):
        """
        Args:
            pred_simcc: (B, K, Nx) + (B, K, Ny) — student SimCC logits.
            teacher_simcc: (B, K, Nx) + (B, K, Ny) — teacher SimCC soft labels.
            target_weight: (B, K) — visibility mask.
        """
        pred_x, pred_y = pred_simcc
        teacher_x, teacher_y = teacher_simcc

        pred_log_x = F.log_softmax(pred_x, dim=-1)
        pred_log_y = F.log_softmax(pred_y, dim=-1)
        teacher_prob_x = F.softmax(teacher_x, dim=-1)
        teacher_prob_y = F.softmax(teacher_y, dim=-1)

        kl_x = F.kl_div(pred_log_x, teacher_prob_x, reduction="none").sum(dim=-1)
        kl_y = F.kl_div(pred_log_y, teacher_prob_y, reduction="none").sum(dim=-1)

        loss = kl_x + kl_y

        if self.use_target_weight and target_weight is not None:
            loss = loss * target_weight.unsqueeze(-1)
            loss = loss.sum() / target_weight.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss * self.loss_weight


class L1CoordinateLoss(nn.Module):
    """L1 loss on decoded coordinates (response distillation)."""

    def __init__(self, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

    @staticmethod
    def decode_simcc(simcc_x, simcc_y):
        """Decode SimCC to coordinates via expected value."""
        B, K, Nx = simcc_x.shape
        Ny = simcc_y.shape[-1]

        x_bins = torch.linspace(0, 1, Nx, device=simcc_x.device).view(1, 1, -1)
        y_bins = torch.linspace(0, 1, Ny, device=simcc_y.device).view(1, 1, -1)

        prob_x = F.softmax(simcc_x, dim=-1)
        prob_y = F.softmax(simcc_y, dim=-1)

        x = (prob_x * x_bins).sum(dim=-1)
        y = (prob_y * y_bins).sum(dim=-1)

        return torch.stack([x, y], dim=-1)

    def forward(self, pred_simcc, teacher_simcc, target_weight=None):
        pred_coords = self.decode_simcc(*pred_simcc)
        teacher_coords = self.decode_simcc(*teacher_simcc)

        loss = F.l1_loss(pred_coords, teacher_coords, reduction="none").sum(dim=-1)

        if target_weight is not None:
            loss = (loss * target_weight).sum() / target_weight.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss * self.loss_weight
