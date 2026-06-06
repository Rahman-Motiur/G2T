from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LHRAAM(nn.Module):
    """Low-to-high resolution attention accumulation."""

    def __init__(self, channels: tuple[int, int, int, int]):
        super().__init__()
        self.filters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 1, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1, 1, 3, padding=1),
                    nn.Sigmoid(),
                )
                for _ in channels
            ]
        )

    def forward(self, features: list[torch.Tensor], attentions: list[torch.Tensor]) -> list[torch.Tensor]:
        accumulated = [None] * len(features)
        running = None
        for i in reversed(range(len(features))):
            current = attentions[i]
            if running is not None:
                running = F.interpolate(running, size=current.shape[-2:], mode="bilinear", align_corners=False)
                current = current + running
            filtered = self.filters[i](current)
            accumulated[i] = features[i] * filtered + features[i]
            running = current
        return accumulated
