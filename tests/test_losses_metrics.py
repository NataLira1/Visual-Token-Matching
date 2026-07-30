import torch

from vtm.losses import segmentation_loss
from vtm.metrics import binary_metrics


def test_invalid_pixels_do_not_change_loss() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    first = segmentation_loss(logits, target, valid)
    changed = target.clone()
    changed[:, :, 1] = 1.0 - changed[:, :, 1]
    second = segmentation_loss(logits, changed, valid)
    torch.testing.assert_close(first, second)


def test_perfect_binary_metrics() -> None:
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    metrics = binary_metrics(logits, target, torch.ones_like(target))
    assert metrics["iou"] > 0.999
    assert metrics["dice"] > 0.999
