from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    target_1h = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_1h, dim=dims)
    denominator = torch.sum(probs + target_1h, dim=dims)
    return 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def ce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 0.3) -> torch.Tensor:
    ce_weight = 1.0 - dice_weight
    return dice_weight * dice_loss(logits, target) + ce_weight * F.cross_entropy(logits, target.long())


def _prediction_combinations(prediction_maps: list[torch.Tensor]) -> list[torch.Tensor]:
    outputs = list(prediction_maps)
    if len(prediction_maps) >= 2:
        outputs.append(prediction_maps[-1] + prediction_maps[-2])
    if len(prediction_maps) >= 3:
        outputs.append(prediction_maps[0] + prediction_maps[1] + prediction_maps[2])
    return outputs


def deep_supervision_loss(prediction_maps: list[torch.Tensor], target: torch.Tensor, dice_weight: float = 0.3) -> torch.Tensor:
    losses = [ce_dice_loss(prediction, target, dice_weight) for prediction in _prediction_combinations(prediction_maps)]
    return torch.stack(losses).mean()


class G2TLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.3):
        super().__init__()
        self.dice_weight = dice_weight

    def forward(self, outputs: dict, target: torch.Tensor) -> dict[str, torch.Tensor]:
        deep = deep_supervision_loss(outputs["prediction_maps"], target, self.dice_weight)
        fused = ce_dice_loss(outputs["logits"], target, self.dice_weight)
        total = 0.5 * deep + 0.5 * fused
        return {"loss": total, "deep_supervision_loss": deep.detach(), "fused_loss": fused.detach()}
