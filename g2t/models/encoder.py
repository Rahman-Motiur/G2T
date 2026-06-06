from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HierarchicalEncoder(nn.Module):
    """Compact four-stage encoder returning feature maps and spatial attentions."""

    def __init__(self, in_channels: int, channels: tuple[int, int, int, int]):
        super().__init__()
        self.stages = nn.ModuleList()
        previous = in_channels
        for i, channel in enumerate(channels):
            self.stages.append(ConvNormAct(previous, channel, stride=1 if i == 0 else 2))
            previous = channel

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features = []
        attentions = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            attentions.append(torch.sigmoid(x.mean(dim=1, keepdim=True)))
        return features, attentions
