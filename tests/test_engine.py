from pathlib import Path

from vtm.data import build_manifest, make_synthetic_taskonomy
from vtm.engine import _classes_with_coverage


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
