from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ConvMSA(nn.Module):
    """Convolutional multi-head self-attention over spatial tokens."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads.")
        self.channels = channels
        self.heads = heads
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        head_dim = c // self.heads
        q = self.q(x).reshape(b, self.heads, head_dim, h * w).transpose(-2, -1)
        k = self.k(x).reshape(b, self.heads, head_dim, h * w)
        v = self.v(x).reshape(b, self.heads, head_dim, h * w).transpose(-2, -1)
        attn = torch.softmax((q @ k) / math.sqrt(head_dim), dim=-1)
        out = (attn @ v).transpose(-2, -1).reshape(b, c, h, w)
        return self.proj(out)


class FeatureAwareDecoderBlock(nn.Module):
    """FAD block with Refine, Expand, and Produce gates."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.up_project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        concat_channels = out_channels + skip_channels
        self.refine_gate = nn.Conv2d(concat_channels, out_channels, 1)
        self.expand_gate = nn.Conv2d(concat_channels, out_channels, 1)
        self.expand_value = nn.Conv2d(concat_channels, out_channels, 3, padding=1)
        self.refined_project = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.produce_value = nn.Conv2d(concat_channels, out_channels, 3, padding=1)
        self.msa = ConvMSA(out_channels, heads)
        self.attention_project = nn.Conv2d(skip_channels, out_channels, 1)
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, previous: torch.Tensor, skip: torch.Tensor, accumulated_attention: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(previous, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        up = self.up_project(up)
        x = torch.cat([up, skip], dim=1)

        refined = up * torch.sigmoid(self.refine_gate(x))
        expanded = torch.sigmoid(self.expand_gate(x)) * F.relu(self.expand_value(x))
        memory = F.relu(self.refined_project(refined)) + expanded

        produce_base = F.relu(self.produce_value(x))
        produce = self.msa(produce_base) + produce_base
        attention = self.attention_project(accumulated_attention)
        return self.output(F.relu(memory) + produce + attention)
