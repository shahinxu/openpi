from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class EncoderConfig:
    backbone: str = "resnet18"
    latent_dim: int = 256
    temporal_layers: int = 4
    temporal_heads: int = 8
    dropout: float = 0.1
    pretrained: bool = True
    freeze_backbone: bool = False
    use_motion_branch: bool = True
    aggregate_last_k: int = 4


class VideoEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.backbone != "resnet18":
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")
        weights = ResNet18_Weights.DEFAULT if cfg.pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [B, 512, 1, 1]
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.frame_proj = nn.Linear(512, cfg.latent_dim)
        self.motion_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.latent_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1025, cfg.latent_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.temporal_heads,
            dim_feedforward=cfg.latent_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=cfg.temporal_layers)
        self.norm = nn.LayerNorm(cfg.latent_dim)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.01)

    def forward(self, x):
        # x: [B, T, C, H, W]
        if x.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got shape={tuple(x.shape)}")

        b, t, c, h, w = x.shape
        frames = x.reshape(b * t, c, h, w)
        feat = self.backbone(frames).reshape(b * t, 512)
        feat = self.frame_proj(feat).reshape(b, t, self.cfg.latent_dim)

        if self.cfg.use_motion_branch:
            if t > 1:
                motion = feat[:, 1:] - feat[:, :-1]
                motion = torch.cat([torch.zeros_like(feat[:, :1]), motion], dim=1)
            else:
                motion = torch.zeros_like(feat)
            feat = feat + self.motion_proj(motion)

        h = feat
        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_emb[:, : t + 1]
        h = self.temporal(h)
        seq_tokens = h[:, 1:]
        tail_k = min(max(self.cfg.aggregate_last_k, 1), t)
        tail_summary = seq_tokens[:, -tail_k:].mean(dim=1)
        z = self.norm(h[:, 0] + tail_summary)
        return z
