from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    lambda_trunk: float = 1.0
    lambda_skeleton: float = 1.0
    lambda_occupancy: float = 0.0
    occupancy_pos_weight_clip_min: float = 1.0
    occupancy_pos_weight_clip_max: float = 200.0


class MultiTaskLoss:
    def __init__(self, cfg: LossConfig) -> None:
        self.cfg = cfg

    def compute(self, preds, targets, update_state: bool = True):
        trunk_loss = F.smooth_l1_loss(preds["trunk6"], targets["trunk6"])
        skel_loss = F.smooth_l1_loss(preds["skeleton153"], targets["skeleton153"])

        occ_targets = targets["occupancy"].float()
        pos = occ_targets.sum()
        neg = float(occ_targets.numel()) - pos
        pos_weight = torch.clamp(
            neg / (pos + 1e-6),
            min=self.cfg.occupancy_pos_weight_clip_min,
            max=self.cfg.occupancy_pos_weight_clip_max,
        )
        occ_loss = F.binary_cross_entropy_with_logits(
            preds["occupancy"],
            occ_targets,
            pos_weight=pos_weight,
        )

        total = (
            self.cfg.lambda_trunk * trunk_loss
            + self.cfg.lambda_skeleton * skel_loss
            + self.cfg.lambda_occupancy * occ_loss
        )

        stats = {
            "loss": float(total.detach().cpu().item()),
            "loss_trunk": float(trunk_loss.detach().cpu().item()),
            "loss_skeleton": float(skel_loss.detach().cpu().item()),
            "loss_occupancy": float(occ_loss.detach().cpu().item()),
        }
        return total, stats
