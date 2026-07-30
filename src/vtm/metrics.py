from __future__ import annotations

from collections.abc import Iterable

import torch


def binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    prediction = (torch.sigmoid(logits) >= threshold) & valid.bool()
    truth = target.bool() & valid.bool()
    valid_bool = valid.bool()
    tp = (prediction & truth).sum().item()
    fp = (prediction & ~truth & valid_bool).sum().item()
    fn = (~prediction & truth).sum().item()
    tn = (~prediction & ~truth & valid_bool).sum().item()
    epsilon = 1e-9
    return {
        "iou": tp / (tp + fp + fn + epsilon),
        "dice": 2 * tp / (2 * tp + fp + fn + epsilon),
        "precision": tp / (tp + fp + epsilon),
        "recall": tp / (tp + fn + epsilon),
        "false_positive_rate": fp / (fp + tn + epsilon),
    }


def mean_metrics(values: Iterable[dict[str, float]]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {}
    return {
        key: sum(value[key] for value in values) / len(values)
        for key in values[0]
    }
