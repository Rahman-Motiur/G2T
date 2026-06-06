from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from g2t import G2T, G2TConfig
from g2t.training import G2TLoss


def main() -> None:
    model = G2T(G2TConfig(in_channels=1, num_classes=3, channels=(16, 32, 64, 128), attention_heads=4))
    images = torch.randn(2, 1, 64, 64)
    masks = torch.randint(0, 3, (2, 64, 64))
    outputs = model(images)
    losses = G2TLoss(dice_weight=0.3)(outputs, masks)
    print("logits:", tuple(outputs["logits"].shape))
    print("prediction maps:", [tuple(x.shape) for x in outputs["prediction_maps"]])
    print("loss:", float(losses["loss"].detach()))


if __name__ == "__main__":
    main()
