from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterator, Sequence

import torch
from torch import nn

from .data import EpisodeSampler, PairRecord, load_pair, select_records
from .losses import segmentation_loss
from .metrics import binary_metrics, mean_metrics
from .model import FrozenFeatureBaseline, VTM
from .utils import default_device, environment_info, seed_everything, write_json


def create_model(config: dict[str, Any], pretrained: bool | None = None) -> VTM:
    model_config = config["model"]
    data_config = config["data"]
    return VTM(
        image_size=int(data_config["image_size"]),
        train_tasks=data_config["train_classes"].keys(),
        backbone=model_config["backbone"],
        pretrained=model_config["pretrained"] if pretrained is None else pretrained,
        feature_blocks=model_config["feature_blocks"],
        label_depth=int(model_config["label_depth"]),
        num_heads=int(model_config["num_heads"]),
        decoder_width=int(model_config["decoder_width"]),
        dropout=float(model_config["dropout"]),
        embed_dim=model_config.get("embed_dim"),
        patch_size=int(model_config.get("patch_size", 16)),
    )


def _episodic_batch(
    episode: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    support_images, support_masks, query_images, query_masks, query_valid = episode
    query_count = query_images.shape[0]
    support_images = support_images.unsqueeze(0).expand(query_count, *support_images.shape)
    support_masks = support_masks.unsqueeze(0).expand(query_count, *support_masks.shape)
    return tuple(
        tensor.to(device)
        for tensor in (
            support_images,
            support_masks,
            query_images,
            query_masks,
            query_valid,
        )
    )


@torch.no_grad()
def validate_meta(
    model: VTM,
    sampler: EpisodeSampler,
    classes: dict[str, int],
    episodes: int,
    device: torch.device,
    limit: int | None,
) -> float:
    model.eval()
    scores = []
    names = list(classes)
    zero_bias = torch.zeros(model.bias_shape, device=device)
    for index in range(episodes):
        name = names[index % len(names)]
        episode = sampler.episode(
            "val",
            classes[name],
            support_size=1,
            query_size=1,
            augment=False,
            limit=limit,
        )
        support_images, support_masks, query_images, query_masks, query_valid = _episodic_batch(
            episode, device
        )
        logits = model(
            query_images,
            support_images,
            support_masks,
            task_bias=zero_bias,
        )
        scores.append(binary_metrics(logits, query_masks, query_valid)["iou"])
    return mean(scores)


def train_meta(
    config: dict[str, Any],
    records: Sequence[PairRecord],
    device: torch.device | None = None,
) -> tuple[VTM, list[dict[str, float]]]:
    seed_everything(int(config["seed"]))
    device = device or default_device()
    data_config, train_config = config["data"], config["train"]
    model = create_model(config).to(device)
    sampler = EpisodeSampler(
        records,
        int(data_config["image_size"]),
        float(data_config["min_positive_fraction"]),
        int(data_config["uncertain_id"]),
        int(config["seed"]),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    amp = bool(train_config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    class_names = list(data_config["train_classes"])
    checkpoint = Path(train_config["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_iou, best_episode = -1.0, 0

    for episode_index in range(1, int(train_config["episodes"]) + 1):
        model.train()
        task = class_names[(episode_index - 1) % len(class_names)]
        class_id = int(data_config["train_classes"][task])
        episode = sampler.episode(
            "train",
            class_id,
            int(train_config["support_size"]),
            int(train_config["query_size"]),
            augment=True,
            limit=int(data_config["max_train_pairs"]),
        )
        support_images, support_masks, query_images, query_masks, query_valid = _episodic_batch(
            episode, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(query_images, support_images, support_masks, task=task)
            loss = segmentation_loss(logits, query_masks, query_valid)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(train_config["gradient_clip"]),
        )
        scaler.step(optimizer)
        scaler.update()
        history.append({"episode": float(episode_index), "loss": float(loss.detach())})

        interval = int(train_config["validation_interval"])
        if episode_index % interval != 0 and episode_index != int(train_config["episodes"]):
            continue
        val_iou = validate_meta(
            model,
            sampler,
            data_config["val_classes"],
            int(train_config["validation_episodes"]),
            device,
            int(data_config["max_val_pairs"]),
        )
        history[-1]["val_iou"] = val_iou
        if val_iou > best_iou:
            best_iou, best_episode = val_iou, episode_index
            torch.save(
                {
                    "model": model.state_dict(),
                    "episode": episode_index,
                    "val_iou": val_iou,
                    "config": config,
                    "environment": environment_info(),
                },
                checkpoint,
            )
        if episode_index - best_episode >= int(train_config["early_stopping_patience"]):
            break

    if not checkpoint.exists():
        raise RuntimeError("O treino terminou sem produzir checkpoint.")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return model, history


@contextmanager
def _frozen_parameters(model: nn.Module) -> Iterator[None]:
    states = {parameter: parameter.requires_grad for parameter in model.parameters()}
    try:
        for parameter in model.parameters():
            parameter.requires_grad = False
        yield
    finally:
        for parameter, state in states.items():
            parameter.requires_grad = state


def _vtm_forward_from_records(
    model: VTM,
    sampler: EpisodeSampler,
    support: Sequence[PairRecord],
    query: Sequence[PairRecord],
    class_id: int,
    bias: torch.Tensor,
    device: torch.device,
    augment: bool,
    return_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor | None] | None]:
    support_images, support_masks, _ = sampler.load_records(support, class_id, augment)
    query_images, query_masks, query_valid = sampler.load_records(query, class_id, augment)
    count = query_images.shape[0]
    support_images = support_images.unsqueeze(0).expand(count, *support_images.shape).to(device)
    support_masks = support_masks.unsqueeze(0).expand(count, *support_masks.shape).to(device)
    result = model(
        query_images.to(device),
        support_images,
        support_masks,
        task_bias=bias,
        return_attention=return_attention,
    )
    if return_attention:
        logits, attention = result
    else:
        logits, attention = result, None
    return logits, query_masks.to(device), query_valid.to(device), attention


def adapt_vtm(
    model: VTM,
    sampler: EpisodeSampler,
    support: Sequence[PairRecord],
    class_id: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with _frozen_parameters(model):
        bias = nn.Parameter(torch.zeros(model.bias_shape, device=device))
        optimizer = torch.optim.Adam([bias], lr=learning_rate)
        for step in range(steps):
            if len(support) == 1:
                sub_support, pseudo_query = support, support
            else:
                query_index = step % len(support)
                pseudo_query = [support[query_index]]
                sub_support = [
                    record for index, record in enumerate(support) if index != query_index
                ]
            optimizer.zero_grad(set_to_none=True)
            logits, target, valid, _ = _vtm_forward_from_records(
                model,
                sampler,
                sub_support,
                pseudo_query,
                class_id,
                bias,
                device,
                augment=True,
            )
            segmentation_loss(logits, target, valid).backward()
            optimizer.step()
        return bias.detach()


def adapt_baseline(
    image_encoder: nn.Module,
    sampler: EpisodeSampler,
    support: Sequence[PairRecord],
    class_id: int,
    steps: int,
    learning_rate: float,
    decoder_width: int,
    device: torch.device,
) -> FrozenFeatureBaseline:
    baseline = FrozenFeatureBaseline(image_encoder, decoder_width).to(device)
    baseline.train()
    optimizer = torch.optim.AdamW(baseline.decoder.parameters(), lr=learning_rate)
    for _ in range(steps):
        images, masks, valid = sampler.load_records(support, class_id, augment=True)
        images, masks, valid = images.to(device), masks.to(device), valid.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = baseline(images)
        segmentation_loss(logits, masks, valid).backward()
        optimizer.step()
    baseline.eval()
    return baseline


@torch.no_grad()
def evaluate_vtm(
    model: VTM,
    sampler: EpisodeSampler,
    support: Sequence[PairRecord],
    queries: Sequence[PairRecord],
    negatives: Sequence[PairRecord],
    class_id: int,
    bias: torch.Tensor,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    values, example = [], {}
    for index, query in enumerate(queries):
        logits, target, valid, attention = _vtm_forward_from_records(
            model,
            sampler,
            support,
            [query],
            class_id,
            bias,
            device,
            augment=False,
            return_attention=index == 0,
        )
        values.append(binary_metrics(logits, target, valid, threshold))
        if index == 0:
            image, _, _ = sampler.load_records([query], class_id, False)
            support_image, support_mask, _ = sampler.load_records(
                [support[0]], class_id, False
            )
            example = {
                "image": image.cpu(),
                "support_image": support_image.cpu(),
                "support_mask": support_mask.cpu(),
                "target": target.cpu(),
                "logits": logits.cpu(),
                "attention": [item.cpu() if item is not None else None for item in attention],
            }
    aggregate = mean_metrics(values)
    if negatives:
        negative_values = []
        for query in negatives:
            logits, target, valid, _ = _vtm_forward_from_records(
                model, sampler, support, [query], class_id, bias, device, False
            )
            negative_values.append(binary_metrics(logits, target, valid, threshold))
        aggregate["false_positive_rate"] = mean(
            value["false_positive_rate"] for value in negative_values
        )
    return aggregate, example


@torch.no_grad()
def evaluate_baseline(
    baseline: FrozenFeatureBaseline,
    sampler: EpisodeSampler,
    queries: Sequence[PairRecord],
    negatives: Sequence[PairRecord],
    class_id: int,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    values, example = [], {}
    for index, query in enumerate(queries):
        image, target, valid = sampler.load_records([query], class_id, False)
        logits = baseline(image.to(device))
        values.append(binary_metrics(logits, target.to(device), valid.to(device), threshold))
        if index == 0:
            example = {
                "image": image.cpu(),
                "target": target.cpu(),
                "logits": logits.cpu(),
            }
    aggregate = mean_metrics(values)
    if negatives:
        negative_values = []
        for query in negatives:
            image, target, valid = sampler.load_records([query], class_id, False)
            logits = baseline(image.to(device))
            negative_values.append(
                binary_metrics(logits, target.to(device), valid.to(device), threshold)
            )
        aggregate["false_positive_rate"] = mean(
            value["false_positive_rate"] for value in negative_values
        )
    return aggregate, example


def _resolve_test_classes(
    config: dict[str, Any],
    records: Sequence[PairRecord],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    data, experiment = config["data"], config["experiment"]
    minimum_fraction = float(data["min_positive_fraction"])
    minimum_supports = int(experiment["min_test_supports"])
    minimum_queries = int(experiment["min_test_queries"])
    selected: dict[str, int] = {}
    changes = []
    candidates = {**data["test_classes"], **data["train_classes"]}
    used_ids = set()
    for requested_name, requested_id in data["test_classes"].items():
        eligible = []
        for name, class_id in candidates.items():
            if int(class_id) in used_ids:
                continue
            supports = select_records(
                records, "train", int(class_id), minimum_fraction
            )
            queries = select_records(records, "test", int(class_id), minimum_fraction)
            if len(supports) >= minimum_supports and len(queries) >= minimum_queries:
                eligible.append((len(supports) + len(queries), name, int(class_id)))
        requested_eligible = next(
            (item for item in eligible if item[2] == int(requested_id)), None
        )
        choice = requested_eligible or (max(eligible) if eligible else None)
        if choice is None:
            raise ValueError(
                f"Nenhuma classe satisfaz {minimum_supports} supports e "
                f"{minimum_queries} queries."
            )
        _, chosen_name, chosen_id = choice
        selected[chosen_name] = chosen_id
        used_ids.add(chosen_id)
        if chosen_id != int(requested_id):
            changes.append(
                {
                    "requested": requested_name,
                    "replacement": chosen_name,
                    "reason": "cobertura insuficiente",
                }
            )
    return selected, changes


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["shots"]))].append(row)
    summary = []
    metrics = ("iou", "dice", "precision", "recall", "false_positive_rate")
    for (method, shots), values in sorted(groups.items()):
        item: dict[str, Any] = {"method": method, "shots": shots}
        for metric in metrics:
            samples = [float(value[metric]) for value in values]
            item[f"{metric}_mean"] = mean(samples)
            item[f"{metric}_std"] = pstdev(samples)
        summary.append(item)
    return summary


def evaluate_experiment(
    config: dict[str, Any],
    records: Sequence[PairRecord],
    model: VTM,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    device = device or default_device()
    seed_everything(int(config["seed"]))
    data, adapt, experiment = config["data"], config["adapt"], config["experiment"]
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = EpisodeSampler(
        records,
        int(data["image_size"]),
        float(data["min_positive_fraction"]),
        int(data["uncertain_id"]),
        int(config["seed"]),
    )
    test_classes, replacements = _resolve_test_classes(config, records)
    rows, examples = [], {}
    for class_name, class_id in test_classes.items():
        supports_pool = sampler.pool("train", class_id)
        query_pool = sampler.pool(
            "test", class_id, int(data["max_test_queries_per_class"])
        )
        negatives = select_records(
            records,
            "test",
            class_id,
            float(data["min_positive_fraction"]),
            limit=min(len(query_pool), 100),
            positive=False,
        )
        for shots in map(int, experiment["shots"]):
            for seed in map(int, experiment["seeds"]):
                rng = random.Random(seed)
                support = rng.sample(supports_pool, shots)
                bias = adapt_vtm(
                    model,
                    sampler,
                    support,
                    class_id,
                    int(adapt["steps"][shots]),
                    float(adapt["learning_rate"]),
                    device,
                )
                vtm_metrics, vtm_example = evaluate_vtm(
                    model,
                    sampler,
                    support,
                    query_pool,
                    negatives,
                    class_id,
                    bias,
                    float(experiment["threshold"]),
                    device,
                )
                baseline = adapt_baseline(
                    model.image_encoder,
                    sampler,
                    support,
                    class_id,
                    int(adapt["baseline_steps"][shots]),
                    float(adapt["baseline_learning_rate"]),
                    int(config["model"]["decoder_width"]),
                    device,
                )
                baseline_metrics, baseline_example = evaluate_baseline(
                    baseline,
                    sampler,
                    query_pool,
                    negatives,
                    class_id,
                    float(experiment["threshold"]),
                    device,
                )
                for method, metrics in (("vtm", vtm_metrics), ("baseline", baseline_metrics)):
                    rows.append(
                        {
                            "method": method,
                            "class": class_name,
                            "class_id": class_id,
                            "shots": shots,
                            "seed": seed,
                            **metrics,
                        }
                    )
                examples[(class_name, shots, seed)] = (vtm_example, baseline_example)

    summary = summarize_results(rows)
    _write_csv(output_dir / "results.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    hypothesis = {}
    for shots in (5, 10):
        relevant = [row for row in summary if int(row["shots"]) == shots]
        if relevant:
            scores = {row["method"]: row["iou_mean"] for row in relevant}
            hypothesis[str(shots)] = {
                "supported": scores.get("vtm", -math.inf)
                > scores.get("baseline", math.inf),
                "vtm_iou": scores.get("vtm"),
                "baseline_iou": scores.get("baseline"),
            }
    write_json(
        output_dir / "hypothesis.json",
        {
            "claim": "VTM supera o baseline em 5-shot e 10-shot.",
            "tests": hypothesis,
            "class_replacements": replacements,
            "environment": environment_info(),
        },
    )
    from .visualize import save_experiment_figures

    save_experiment_figures(output_dir, examples)
    return rows


def load_checkpoint(config: dict[str, Any], device: torch.device | None = None) -> VTM:
    device = device or default_device()
    model = create_model(config, pretrained=False).to(device)
    payload = torch.load(
        config["train"]["checkpoint"],
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def save_history(path: str | Path, history: Sequence[dict[str, float]]) -> None:
    write_json(path, list(history))
