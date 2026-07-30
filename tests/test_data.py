from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vtm.data import (
    build_manifest,
    load_pair,
    make_synthetic_taskonomy,
)


def test_pairing_and_nearest_mask_resize(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_synthetic_taskonomy(root, image_size=24)
    records = build_manifest(root, tmp_path / "manifest.json", range(3, 8))
    assert len(records) == 36
    assert {record.split for record in records} == {"train", "val", "test"}
    image, target, valid = load_pair(records[0], 3, image_size=32)
    assert image.shape == (3, 32, 32)
    assert target.shape == valid.shape == (1, 32, 32)
    assert set(torch.unique(target).tolist()) <= {0.0, 1.0}


def test_uncertain_pixels_are_invalid(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_synthetic_taskonomy(root, image_size=16)
    records = build_manifest(root, tmp_path / "manifest.json", [3])
    _, _, valid = load_pair(records[0], 3, image_size=16, uncertain_id=1)
    assert valid.sum() < valid.numel()


def test_palette_mask_preserves_class_ids(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_synthetic_taskonomy(root, image_size=16)
    records = build_manifest(root, tmp_path / "manifest.json", [3])
    record = records[0]
    values = np.ones((16, 16), dtype=np.uint8)
    values[2:10, 2:10] = 3
    palette_mask = Image.fromarray(values, mode="P")
    palette = [0, 0, 0] * 256
    palette[9:12] = [255, 0, 0]
    palette_mask.putpalette(palette)
    palette_mask.save(record.mask)
    _, target, _ = load_pair(record, 3, image_size=16)
    assert target.sum().item() == 64
