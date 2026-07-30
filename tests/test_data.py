from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vtm.data import (
    build_manifest,
    filter_readable_records,
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


def test_image_hash_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_synthetic_taskonomy(root, image_size=16)
    first = build_manifest(
        root,
        tmp_path / "first.json",
        [3],
        split_strategy="image_hash",
        split_ratios={"train": 0.70, "val": 0.15, "test": 0.15},
        split_seed=7,
    )
    second = build_manifest(
        root,
        tmp_path / "second.json",
        [3],
        split_strategy="image_hash",
        split_ratios={"train": 0.70, "val": 0.15, "test": 0.15},
        split_seed=7,
    )
    assert [(record.key, record.split) for record in first] == [
        (record.key, record.split) for record in second
    ]
    keys_by_split = {
        split: {record.key for record in first if record.split == split}
        for split in ("train", "val", "test")
    }
    assert all(keys_by_split.values())
    assert keys_by_split["train"].isdisjoint(keys_by_split["val"])
    assert keys_by_split["train"].isdisjoint(keys_by_split["test"])
    assert keys_by_split["val"].isdisjoint(keys_by_split["test"])
    assert {record.building for record in first} == {
        "hanson",
        "wiconisco",
        "muleshoe",
    }


def test_corrupted_rgb_is_excluded(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_synthetic_taskonomy(root, image_size=16)
    records = build_manifest(root, tmp_path / "manifest.json", [3])
    Path(records[0].rgb).write_bytes(b"not-an-image")

    readable, failures = filter_readable_records(records)

    assert len(readable) == len(records) - 1
    assert failures == [
        {
            "key": records[0].key,
            "building": records[0].building,
            "domain": "rgb",
            "path": records[0].rgb,
            "error": "UnidentifiedImageError: cannot identify image file "
            f"'{records[0].rgb}'",
        }
    ]


def test_rgb_is_aligned_to_lower_resolution_mask(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_synthetic_taskonomy(root, image_size=16)
    records = build_manifest(root, tmp_path / "manifest.json", [3])
    record = records[0]
    rgb = Image.open(record.rgb).convert("RGB").resize((32, 32))
    rgb.save(record.rgb)

    image, target, valid = load_pair(record, 3, image_size=24)

    assert image.shape == (3, 24, 24)
    assert target.shape == valid.shape == (1, 24, 24)
    assert set(torch.unique(target).tolist()) <= {0.0, 1.0}
