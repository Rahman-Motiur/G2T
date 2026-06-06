import torch

from g2t import G2T, G2TConfig
from g2t.training import G2TLoss


def test_g2t_forward_and_loss():
    model = G2T(G2TConfig(in_channels=1, num_classes=3, channels=(8, 16, 32, 64), attention_heads=4))
    images = torch.randn(1, 1, 32, 32)
    masks = torch.randint(0, 3, (1, 32, 32))
    outputs = model(images)
    assert outputs["logits"].shape == (1, 3, 32, 32)
    assert len(outputs["prediction_maps"]) == 3
    losses = G2TLoss()(outputs, masks)
    assert torch.isfinite(losses["loss"])
