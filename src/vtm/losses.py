from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denominator = valid.sum().clamp_min(1.0)
    return (pixel_loss * valid).sum() / denominator


def masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits) * valid
    target = target * valid
    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * masked_bce(logits, target, valid) + 0.5 * masked_dice_loss(
        logits, target, valid
    )
