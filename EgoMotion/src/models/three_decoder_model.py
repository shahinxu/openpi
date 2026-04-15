from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from src.decoders.heads import DecoderConfig, OccupancyDecoder, SkeletonDecoder, TrunkDecoder
from src.models.encoder import EncoderConfig, VideoEncoder


@dataclass
class ThreeDecoderModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


class ThreeDecoderModel(nn.Module):
    def __init__(self, cfg: ThreeDecoderModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = VideoEncoder(cfg.encoder)
        self.trunk_head = TrunkDecoder(cfg.decoder)
        self.skel_head = SkeletonDecoder(cfg.decoder)
        self.occ_head = OccupancyDecoder(cfg.decoder)

    def forward(self, video):
        z = self.encoder.forward(video)
        trunk6 = self.trunk_head.forward(z)
        skeleton153 = self.skel_head.forward(z)
        occupancy = self.occ_head.forward(z)
        return {
            "trunk6": trunk6,
            "skeleton153": skeleton153,
            "occupancy": occupancy,
        }
