from pathlib import Path

import pytest

from vtm.config import smoke_config
from vtm.data import build_manifest, make_synthetic_taskonomy
from vtm.engine import _classes_with_coverage, _resolve_test_classes


def test_classes_with_insufficient_coverage_are_excluded(tmp_path: Path) -> None:
    root = tmp_path / "data"
    make_synthetic_taskonomy(root, image_size=16)
    records = build_manifest(root, tmp_path / "manifest.json", [3, 99])

    eligible, excluded = _classes_with_coverage(
        records,
        {"present": 3, "absent": 99},
        split="train",
        minimum_fraction=0.01,
        required=2,
    )

    assert eligible == {"present": 3}
    assert excluded == {"absent": 0}


def test_test_class_is_never_replaced_by_training_class(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    config["data"]["test_classes"] = {"unseen_absent": 99}
    make_synthetic_taskonomy(config["data"]["root"], image_size=16)
    records = build_manifest(
        config["data"]["root"],
        config["data"]["manifest"],
        [3, 4, 99],
    )

    with pytest.raises(ValueError, match="meta-teste"):
        _resolve_test_classes(config, records)
