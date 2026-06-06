from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import HierarchicalEncoder
from .fad import FeatureAwareDecoderBlock
from .lh_raam import LHRAAM


@dataclass
class G2TConfig:
    in_channels: int = 1
    num_classes: int = 4
    channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    attention_heads: int = 4
    freeze_encoder: bool = False


class BottleneckBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.d3 = nn.Conv2d(channels, channels * 2, 3, padding=3, dilation=3)
        self.norm = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.norm(self.d1(x)))
        x2 = F.relu(self.d2(x1))
        x3 = F.relu(self.d3(x2))
        return self.drop(x3)


class G2T(nn.Module):
    def __init__(self, config: G2TConfig):
        super().__init__()
        self.config = config
        c1, c2, c3, c4 = config.channels
        self.encoder = HierarchicalEncoder(config.in_channels, config.channels)
        if config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.lh_raam = LHRAAM(config.channels)
        self.bottleneck = BottleneckBlock(c4)
        self.decoder4 = FeatureAwareDecoderBlock(c4 * 2, c4, c4, config.attention_heads)
        self.decoder3 = FeatureAwareDecoderBlock(c4, c3, c3, config.attention_heads)
        self.decoder2 = FeatureAwareDecoderBlock(c3, c2, c2, config.attention_heads)
        self.decoder1 = FeatureAwareDecoderBlock(c2, c1, c1, config.attention_heads)
        self.head3 = nn.Conv2d(c3, config.num_classes, 1)
        self.head2 = nn.Conv2d(c2, config.num_classes, 1)
        self.head1 = nn.Conv2d(c1, config.num_classes, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        image_size = x.shape[-2:]
        features, attentions = self.encoder(x)
        accumulated = self.lh_raam(features, attentions)
        x4 = self.bottleneck(features[3])
        d4 = self.decoder4(x4, features[3], accumulated[3])
        d3 = self.decoder3(d4, features[2], accumulated[2])
        d2 = self.decoder2(d3, features[1], accumulated[1])
        d1 = self.decoder1(d2, features[0], accumulated[0])

        p3 = F.interpolate(self.head3(d3), size=image_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.head2(d2), size=image_size, mode="bilinear", align_corners=False)
        p1 = F.interpolate(self.head1(d1), size=image_size, mode="bilinear", align_corners=False)
        fused = p1 + p2 + p3
        return {"logits": fused, "prediction_maps": [p1, p2, p3], "features": [d1, d2, d3, d4]}
