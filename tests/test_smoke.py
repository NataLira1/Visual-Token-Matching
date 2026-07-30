from pathlib import Path

from vtm.cli import command_smoke


def test_end_to_end_smoke(tmp_path: Path) -> None:
    command_smoke(tmp_path / "smoke")
    assert (tmp_path / "smoke" / "vtm_best.pt").exists()
    assert (tmp_path / "smoke" / "evaluation" / "results.csv").exists()
    assert (tmp_path / "smoke" / "evaluation" / "hypothesis.json").exists()
