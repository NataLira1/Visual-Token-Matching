from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    image = tensor[0] * IMAGENET_STD + IMAGENET_MEAN
    return image.permute(1, 2, 0).clamp(0, 1).numpy()


def save_experiment_figures(
    output_dir: Path,
    examples: dict[tuple[str, int, int], tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    panel_dir, attention_dir = output_dir / "panels", output_dir / "attention"
    panel_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)
    for (class_name, shots, seed), (vtm, baseline) in examples.items():
        figure, axes = plt.subplots(1, 6, figsize=(18, 3))
        axes[0].imshow(_rgb(vtm["image"]))
        axes[0].set_title("Query RGB")
        axes[1].imshow(_rgb(vtm["support_image"]))
        axes[1].set_title("Support RGB")
        axes[2].imshow(vtm["support_mask"][0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Support mask")
        axes[3].imshow(vtm["target"][0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Ground truth")
        axes[4].imshow(torch.sigmoid(baseline["logits"])[0, 0].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[4].set_title("Baseline")
        axes[5].imshow(torch.sigmoid(vtm["logits"])[0, 0].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[5].set_title("VTM")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        stem = f"{class_name}_{shots}shot_seed{seed}"
        figure.savefig(panel_dir / f"{stem}.png", dpi=150)
        plt.close(figure)

        attention = vtm.get("attention", [])
        if attention and attention[-1] is not None:
            weights = attention[-1][0, 0]
            query_index = weights.shape[0] // 2
            support_weights = weights[query_index]
            grid = int(round((support_weights.numel() / shots) ** 0.5))
            if grid * grid * shots == support_weights.numel():
                maps = support_weights.reshape(shots, grid, grid)
                figure, axes = plt.subplots(1, shots, figsize=(3 * shots, 3), squeeze=False)
                for index in range(shots):
                    axes[0, index].imshow(maps[index].numpy(), cmap="viridis")
                    axes[0, index].set_title(f"Support {index + 1}")
                    axes[0, index].axis("off")
                figure.tight_layout()
                figure.savefig(attention_dir / f"{stem}.png", dpi=150)
                plt.close(figure)


def save_training_curves(path: Path, history: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    episodes = [int(item["episode"]) for item in history]
    losses = [item["loss"] for item in history]
    validation = [
        (int(item["episode"]), item["val_iou"])
        for item in history
        if "val_iou" in item
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(episodes, losses)
    axes[0].set(title="Meta-training loss", xlabel="Episódio", ylabel="Loss")
    if validation:
        axes[1].plot(
            [item[0] for item in validation],
            [item[1] for item in validation],
            marker="o",
        )
    axes[1].set(title="Meta-validation", xlabel="Episódio", ylabel="IoU")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
