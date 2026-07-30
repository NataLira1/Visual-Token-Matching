from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Carrega o YAML e resolve caminhos relativos a partir do diretório atual."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("A configuração deve ser um mapeamento YAML.")
    return config


def smoke_config(output_dir: str | Path) -> dict[str, Any]:
    """Configuração pequena, usada pelo CLI e pelos testes."""
    output = Path(output_dir)
    return {
        "seed": 0,
        "data": {
            "root": str(output / "synthetic_data"),
            "manifest": str(output / "manifest.json"),
            "image_size": 32,
            "min_positive_fraction": 0.01,
            "max_train_pairs": 32,
            "max_val_pairs": 16,
            "max_test_queries_per_class": 8,
            "uncertain_id": 0,
            "train_classes": {"shape_a": 3, "shape_b": 4},
            "val_classes": {"shape_c": 5},
            "test_classes": {"shape_d": 6, "shape_e": 7},
        },
        "model": {
            "backbone": "tiny_cnn",
            "pretrained": False,
            "feature_blocks": [0, 1, 2, 3],
            "embed_dim": 32,
            "patch_size": 4,
            "label_depth": 4,
            "num_heads": 4,
            "decoder_width": 32,
            "dropout": 0.0,
        },
        "train": {
            "episodes": 2,
            "support_size": 2,
            "query_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "validation_interval": 1,
            "validation_episodes": 1,
            "early_stopping_patience": 10,
            "mixed_precision": False,
            "checkpoint": str(output / "vtm_best.pt"),
        },
        "adapt": {
            "learning_rate": 0.01,
            "steps": {1: 1},
            "baseline_learning_rate": 0.01,
            "baseline_steps": {1: 1},
        },
        "experiment": {
            "shots": [1],
            "seeds": [0],
            "output_dir": str(output / "evaluation"),
            "threshold": 0.5,
            "min_test_supports": 1,
            "min_test_queries": 1,
        },
    }


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    updated = deepcopy(config)
    for dotted_key, value in overrides.items():
        cursor = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return updated
