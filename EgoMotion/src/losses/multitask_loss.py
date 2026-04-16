from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    lambda_trunk: float = 1.0
    lambda_skeleton: float = 1.0
    lambda_occupancy: float = 2.0
    occupancy_pos_weight_clip_min: float = 1.0
    occupancy_pos_weight_clip_max: float = 200.0
    normalize_components: bool = True
    norm_ema_decay: float = 0.99
    norm_eps: float = 1e-6


class MultiTaskLoss:
    def __init__(self, cfg: LossConfig) -> None:
        self.cfg = cfg
        self._trunk_scale_ema: float | None = None
        self._skeleton_scale_ema: float | None = None
        self._occupancy_scale_ema: float | None = None

    def _update_ema(self, current_ema: float | None, new_value: float) -> float:
        if current_ema is None:
            return new_value
        decay = self.cfg.norm_ema_decay
        return decay * current_ema + (1.0 - decay) * new_value

    def _normalize_component(
        self,
        loss_tensor: torch.Tensor,
        current_ema: float | None,
        update_state: bool,
    ) -> tuple[torch.Tensor, float | None]:
        detached_val = float(loss_tensor.detach().cpu().item())
        if current_ema is None:
            scale_ema = detached_val
        elif update_state:
            scale_ema = self._update_ema(current_ema, detached_val)
        else:
            scale_ema = current_ema
        normalized = loss_tensor / max(scale_ema, self.cfg.norm_eps)
        return normalized, (scale_ema if update_state else current_ema)

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

        if self.cfg.normalize_components:
            trunk_term, new_trunk_ema = self._normalize_component(trunk_loss, self._trunk_scale_ema, update_state)
            skel_term, new_skel_ema = self._normalize_component(skel_loss, self._skeleton_scale_ema, update_state)
            occ_term, new_occ_ema = self._normalize_component(occ_loss, self._occupancy_scale_ema, update_state)
            if update_state:
                self._trunk_scale_ema = new_trunk_ema
                self._skeleton_scale_ema = new_skel_ema
                self._occupancy_scale_ema = new_occ_ema
        else:
            trunk_term = trunk_loss
            skel_term = skel_loss
            occ_term = occ_loss

        total = (
            self.cfg.lambda_trunk * trunk_term
            + self.cfg.lambda_skeleton * skel_term
            + self.cfg.lambda_occupancy * occ_term
        )

        stats = {
            "loss": float(total.detach().cpu().item()),
            "loss_trunk": float(trunk_term.detach().cpu().item()),
            "loss_skeleton": float(skel_term.detach().cpu().item()),
            "loss_occupancy": float(occ_term.detach().cpu().item()),
            "raw_loss_trunk": float(trunk_loss.detach().cpu().item()),
            "raw_loss_skeleton": float(skel_loss.detach().cpu().item()),
            "raw_loss_occupancy": float(occ_loss.detach().cpu().item()),
            "occ_pos_weight": float(pos_weight.detach().cpu().item()),
        }
        if self.cfg.normalize_components:
            stats.update(
                {
                    "scale_trunk": float(self._trunk_scale_ema),
                    "scale_skeleton": float(self._skeleton_scale_ema),
                    "scale_occupancy": float(self._occupancy_scale_ema),
                }
            )
        return total, stats
