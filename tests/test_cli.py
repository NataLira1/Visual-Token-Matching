from pathlib import Path

import pytest

from vtm.cli import PrerequisiteError, command_evaluate, command_prepare, command_train
from vtm.config import smoke_config


def test_prepare_reports_missing_dataset(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    with pytest.raises(PrerequisiteError, match="data.root"):
        command_prepare(config)


def test_train_reports_missing_manifest(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    with pytest.raises(PrerequisiteError, match="Execute primeiro"):
        command_train(config)


def test_evaluate_reports_missing_checkpoint(tmp_path: Path) -> None:
    config = smoke_config(tmp_path)
    manifest = Path(config["data"]["manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(PrerequisiteError, match="Checkpoint"):
        command_evaluate(config)
