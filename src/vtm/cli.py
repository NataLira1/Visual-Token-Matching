from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

from .config import load_config, smoke_config
from .data import build_manifest, load_manifest, make_synthetic_taskonomy, select_records
from .engine import (
    evaluate_experiment,
    load_checkpoint,
    save_history,
    train_meta,
)
from .utils import default_device, environment_info, write_json
from .visualize import save_training_curves


class PrerequisiteError(RuntimeError):
    """Erro acionável para uma etapa anterior ainda não executada."""


def _class_ids(config: dict[str, Any]) -> set[int]:
    data = config["data"]
    return {
        int(class_id)
        for group in ("train_classes", "val_classes", "test_classes")
        for class_id in data[group].values()
    }


def _require_data_root(config: dict[str, Any]) -> Path:
    root = Path(config["data"]["root"]).expanduser()
    if not root.exists():
        raise PrerequisiteError(
            f"Diretório Taskonomy não encontrado: {root}\n"
            "Baixe os domínios rgb e segment_semantic ou ajuste data.root no YAML.\n"
            "Para testar o código sem o dataset, execute:\n"
            "  vtm-taskonomy smoke --output-dir outputs/smoke"
        )
    return root


def _require_manifest(config: dict[str, Any]) -> Path:
    manifest = Path(config["data"]["manifest"]).expanduser()
    if not manifest.exists():
        raise PrerequisiteError(
            f"Manifest não encontrado: {manifest}\n"
            "Execute primeiro, e aguarde a conclusão de:\n"
            "  vtm-taskonomy prepare --config configs/taskonomy_vtm.yaml"
        )
    return manifest


def _require_checkpoint(config: dict[str, Any]) -> Path:
    checkpoint = Path(config["train"]["checkpoint"]).expanduser()
    if not checkpoint.exists():
        raise PrerequisiteError(
            f"Checkpoint não encontrado: {checkpoint}\n"
            "Execute primeiro, e aguarde a conclusão de:\n"
            "  vtm-taskonomy train --config configs/taskonomy_vtm.yaml"
        )
    return checkpoint


def command_prepare(config: dict[str, Any]) -> None:
    _require_data_root(config)
    records = build_manifest(
        config["data"]["root"],
        config["data"]["manifest"],
        _class_ids(config),
        split_strategy=config["data"].get("split_strategy", "building"),
        split_ratios=config["data"].get("split_ratios"),
        split_seed=int(config["data"].get("split_seed", config["seed"])),
    )
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    print(f"Manifest criado com {len(records)} pares: {counts}")
    minimum = float(config["data"]["min_positive_fraction"])
    for group, split in (
        ("train_classes", "train"),
        ("val_classes", "val"),
        ("test_classes", "test"),
    ):
        coverage = {
            name: len(select_records(records, split, int(class_id), minimum))
            for name, class_id in config["data"][group].items()
        }
        print(f"Cobertura positiva em {split}: {coverage}")


def command_train(config: dict[str, Any]) -> None:
    _require_manifest(config)
    records = load_manifest(config["data"]["manifest"])
    device = default_device()
    print(f"Treinando em {device}: {environment_info()}")
    _, history = train_meta(config, records, device)
    history_path = Path(config["train"]["checkpoint"]).with_name("training_history.json")
    save_history(history_path, history)
    save_training_curves(history_path.with_suffix(".png"), history)
    print(f"Checkpoint: {config['train']['checkpoint']}")


def command_evaluate(config: dict[str, Any]) -> None:
    _require_manifest(config)
    _require_checkpoint(config)
    records = load_manifest(config["data"]["manifest"])
    device = default_device()
    model = load_checkpoint(config, device)
    rows = evaluate_experiment(config, records, model, device)
    print(f"Avaliação concluída: {len(rows)} linhas em {config['experiment']['output_dir']}")


def command_smoke(output_dir: str | Path) -> None:
    config = smoke_config(output_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "smoke_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    make_synthetic_taskonomy(config["data"]["root"], int(config["data"]["image_size"]))
    command_prepare(config)
    command_train(config)
    command_evaluate(config)
    write_json(output / "environment.json", environment_info())
    print(f"Smoke test completo em {output.resolve()}")


def command_doctor(config: dict[str, Any]) -> None:
    paths = {
        "dados": Path(config["data"]["root"]).expanduser(),
        "manifest": Path(config["data"]["manifest"]).expanduser(),
        "checkpoint": Path(config["train"]["checkpoint"]).expanduser(),
    }
    for label, path in paths.items():
        status = "OK" if path.exists() else "AUSENTE"
        print(f"[{status:7}] {label:10} {path}")
    print(f"[INFO   ] ambiente   {environment_info()}")
    if not paths["dados"].exists():
        print(
            "\nPróxima etapa: baixe Taskonomy em data/taskonomy ou altere "
            "data.root no arquivo de configuração."
        )
    elif not paths["manifest"].exists():
        print("\nPróxima etapa: execute `vtm-taskonomy prepare --config ...`.")
    elif not paths["checkpoint"].exists():
        print("\nPróxima etapa: execute `vtm-taskonomy train --config ...`.")
    else:
        print("\nAmbiente pronto para `vtm-taskonomy evaluate --config ...`.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimento VTM no Taskonomy")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "prepare", "train", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, help="Caminho para o YAML.")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", default="outputs/smoke")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        command_smoke(args.output_dir)
        return
    config = load_config(args.config)
    try:
        {
            "doctor": command_doctor,
            "prepare": command_prepare,
            "train": command_train,
            "evaluate": command_evaluate,
        }[args.command](config)
    except PrerequisiteError as error:
        print(f"Erro de pré-requisito:\n{error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
