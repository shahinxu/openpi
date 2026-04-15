from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class DecoderConfig:
    latent_dim: int = 256
    hidden_dim: int = 512
    occ_shape: tuple[int, int, int] = (7, 15, 15)


class TrunkDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 6),
        )

    def forward(self, z):
        return self.net(z)


class SkeletonDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 153),
        )

    def forward(self, z):
        return self.net(z)


class OccupancyDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        out_dim = int(cfg.occ_shape[0] * cfg.occ_shape[1] * cfg.occ_shape[2])
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, out_dim),
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, self.cfg.occ_shape[0], self.cfg.occ_shape[1], self.cfg.occ_shape[2])
