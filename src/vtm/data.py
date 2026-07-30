from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageDraw

from .constants import BUILDING_TO_SPLIT
from .utils import write_json

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DOMAIN_PATTERN = re.compile(
    r"(?:_domain_)?(?:segment_?semantic|segmentsemantic|rgb)", re.IGNORECASE
)
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class PairRecord:
    key: str
    building: str
    split: str
    rgb: str
    mask: str
    coverage: dict[str, float]

    @classmethod
    def from_dict(cls, value: dict) -> "PairRecord":
        return cls(
            key=value["key"],
            building=value["building"],
            split=value["split"],
            rgb=value["rgb"],
            mask=value["mask"],
            coverage={str(k): float(v) for k, v in value["coverage"].items()},
        )


def _file_domain(path: Path) -> str | None:
    lowered_parts = [part.lower() for part in path.parts]
    if "segment_semantic" in lowered_parts or "segmentsemantic" in lowered_parts:
        return "mask"
    if "rgb" in lowered_parts:
        return "rgb"
    name = path.stem.lower()
    if "segment_semantic" in name or "segmentsemantic" in name:
        return "mask"
    if re.search(r"(?:^|_)rgb(?:_|$)", name):
        return "rgb"
    return None


def _building(path: Path) -> str | None:
    lower_parts = [part.lower() for part in path.parts]
    for part in lower_parts:
        if part in BUILDING_TO_SPLIT:
            return part
    lowered_name = path.name.lower()
    for candidate in BUILDING_TO_SPLIT:
        if lowered_name.startswith(candidate + "_"):
            return candidate
    return None


def _pair_key(path: Path, building: str) -> str:
    clean = DOMAIN_PATTERN.sub("", path.stem.lower())
    clean = re.sub(r"[_-]+", "_", clean).strip("_")
    if clean.startswith(building + "_"):
        clean = clean[len(building) + 1 :]
    return f"{building}/{clean}"


def discover_pairs(root: str | Path) -> list[tuple[str, str, Path, Path]]:
    """Descobre layouts Taskonomy originais, Omnidata e o layout achatado do VTM."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Diretório Taskonomy não encontrado: {root}")
    domains: dict[tuple[str, str], dict[str, Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        domain = _file_domain(path)
        building = _building(path)
        if domain is None or building is None:
            continue
        key = _pair_key(path, building)
        domains.setdefault((building, key), {})[domain] = path

    pairs = []
    for (building, key), values in domains.items():
        if "rgb" in values and "mask" in values:
            pairs.append((key, building, values["rgb"], values["mask"]))
    if not pairs:
        raise RuntimeError(
            "Nenhum par RGB/máscara foi encontrado. Confira data.root e se os "
            "domínios rgb e segment_semantic foram baixados."
        )
    return sorted(pairs, key=lambda item: item[0])


def _mask_coverage(path: Path, class_ids: Iterable[int]) -> dict[str, float]:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        if not np.array_equal(mask[..., 0], mask[..., -1]):
            raise ValueError(
                f"Máscara semântica RGB não codifica IDs diretamente: {path}"
            )
        mask = mask[..., 0]
    denominator = float(mask.size)
    return {str(class_id): float((mask == class_id).sum() / denominator) for class_id in class_ids}


def build_manifest(
    root: str | Path,
    output: str | Path,
    class_ids: Iterable[int],
    split_strategy: str = "building",
    split_ratios: dict[str, float] | None = None,
    split_seed: int = 0,
) -> list[PairRecord]:
    pairs = discover_pairs(root)
    if split_strategy == "building":
        assigned_splits = {
            key: BUILDING_TO_SPLIT[building] for key, building, _, _ in pairs
        }
    elif split_strategy == "image_hash":
        ratios = split_ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
        expected = {"train", "val", "test"}
        if set(ratios) != expected or any(float(value) <= 0 for value in ratios.values()):
            raise ValueError("split_ratios deve conter train/val/test com valores positivos.")
        total = sum(float(value) for value in ratios.values())
        normalized = {name: float(value) / total for name, value in ratios.items()}
        ranked = sorted(
            pairs,
            key=lambda pair: hashlib.sha256(
                f"{split_seed}:{pair[0]}".encode("utf-8")
            ).digest(),
        )
        count = len(ranked)
        train_end = round(count * normalized["train"])
        val_end = train_end + round(count * normalized["val"])
        assigned_splits = {}
        for index, (key, _, _, _) in enumerate(ranked):
            assigned_splits[key] = (
                "train" if index < train_end else "val" if index < val_end else "test"
            )
    else:
        raise ValueError(
            f"split_strategy desconhecida: {split_strategy!r}; "
            "use 'building' ou 'image_hash'."
        )

    records = []
    for key, building, rgb, mask in pairs:
        records.append(
            PairRecord(
                key=key,
                building=building,
                split=assigned_splits[key],
                rgb=str(rgb.resolve()),
                mask=str(mask.resolve()),
                coverage=_mask_coverage(mask, class_ids),
            )
        )
    write_json(
        output,
        {
            "root": str(Path(root).resolve()),
            "split_strategy": split_strategy,
            "split_ratios": split_ratios,
            "split_seed": split_seed,
            "records": [record.__dict__ for record in records],
        },
    )
    return records


def load_manifest(path: str | Path) -> list[PairRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [PairRecord.from_dict(record) for record in payload["records"]]


def filter_readable_records(
    records: Sequence[PairRecord],
) -> tuple[list[PairRecord], list[dict[str, str]]]:
    """Remove pares corrompidos sem aceitar imagens parcialmente decodificadas."""
    readable, failures = [], []
    for record in records:
        failure = None
        for domain, path in (("rgb", record.rgb), ("mask", record.mask)):
            try:
                with Image.open(path) as image:
                    image.load()
            except (OSError, ValueError, SyntaxError) as error:
                failure = {
                    "key": record.key,
                    "building": record.building,
                    "domain": domain,
                    "path": path,
                    "error": f"{type(error).__name__}: {error}",
                }
                break
        if failure is None:
            readable.append(record)
        else:
            failures.append(failure)
    return readable, failures


def select_records(
    records: Sequence[PairRecord],
    split: str,
    class_id: int,
    min_fraction: float,
    limit: int | None = None,
    positive: bool = True,
) -> list[PairRecord]:
    selected = [
        record
        for record in records
        if record.split == split
        and (
            record.coverage.get(str(class_id), 0.0) >= min_fraction
            if positive
            else record.coverage.get(str(class_id), 0.0) == 0.0
        )
    ]
    return selected[:limit] if limit is not None else selected


def _joint_transform(
    image: Image.Image,
    mask: Image.Image,
    size: int,
    augment: bool,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    if augment:
        width, height = image.size
        scale = rng.uniform(0.85, 1.0)
        crop_w, crop_h = max(1, int(width * scale)), max(1, int(height * scale))
        left = rng.randint(0, max(0, width - crop_w))
        top = rng.randint(0, max(0, height - crop_h))
        box = (left, top, left + crop_w, top + crop_h)
        image, mask = image.crop(box), mask.crop(box)
        if rng.random() < 0.5:
            image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    mask = mask.resize((size, size), Image.Resampling.NEAREST)
    if augment:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
    return image, mask


def load_pair(
    record: PairRecord,
    class_id: int,
    image_size: int,
    uncertain_id: int = 0,
    augment: bool = False,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = rng or random.Random(0)
    image = Image.open(record.rgb).convert("RGB")
    raw_mask = Image.open(record.mask)
    # Os arquivos Taskonomy/Omnidata podem armazenar RGB em 512² e máscaras em
    # 256². Eles representam o mesmo campo de visão, portanto alinhamos o RGB à
    # grade da máscara antes de qualquer crop/flip sincronizado. A máscara nunca
    # usa interpolação contínua, preservando os IDs inteiros das classes.
    if image.size != raw_mask.size:
        image = image.resize(raw_mask.size, Image.Resampling.BILINEAR)
    image, raw_mask = _joint_transform(image, raw_mask, image_size, augment, rng)

    image_array = np.array(image, dtype=np.float32, copy=True) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

    mask_array = np.array(raw_mask, dtype=np.int64, copy=True)
    if mask_array.ndim == 3:
        if not np.array_equal(mask_array[..., 0], mask_array[..., -1]):
            raise ValueError(
                f"Máscara RGB não contém IDs repetidos por canal: {record.mask}"
            )
        mask_array = mask_array[..., 0]
    target = torch.from_numpy((mask_array == class_id).astype(np.float32))[None]
    valid = torch.from_numpy((mask_array != uncertain_id).astype(np.float32))[None]
    return image_tensor, target, valid


class EpisodeSampler:
    def __init__(
        self,
        records: Sequence[PairRecord],
        image_size: int,
        min_positive_fraction: float,
        uncertain_id: int,
        seed: int = 0,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.min_positive_fraction = min_positive_fraction
        self.uncertain_id = uncertain_id
        self.rng = random.Random(seed)

    def pool(self, split: str, class_id: int, limit: int | None = None) -> list[PairRecord]:
        return select_records(
            self.records,
            split,
            class_id,
            self.min_positive_fraction,
            limit,
        )

    def load_records(
        self,
        records: Sequence[PairRecord],
        class_id: int,
        augment: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loaded = [
            load_pair(
                record,
                class_id,
                self.image_size,
                self.uncertain_id,
                augment,
                self.rng,
            )
            for record in records
        ]
        images, masks, valid = zip(*loaded)
        return torch.stack(images), torch.stack(masks), torch.stack(valid)

    def episode(
        self,
        split: str,
        class_id: int,
        support_size: int,
        query_size: int,
        augment: bool,
        limit: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pool = self.pool(split, class_id, limit)
        required = support_size + query_size
        if len(pool) < required:
            raise ValueError(
                f"Classe {class_id} tem {len(pool)} exemplos positivos em {split}; "
                f"são necessários {required}."
            )
        chosen = self.rng.sample(pool, required)
        support = chosen[:support_size]
        query = chosen[support_size:]
        support_images, support_masks, _ = self.load_records(support, class_id, augment)
        query_images, query_masks, query_valid = self.load_records(query, class_id, augment)
        return support_images, support_masks, query_images, query_masks, query_valid


def make_synthetic_taskonomy(root: str | Path, image_size: int = 32) -> None:
    """Cria um Taskonomy mínimo que exercita descoberta, treino e avaliação."""
    root = Path(root)
    specs = (
        ("hanson", "train"),
        ("wiconisco", "val"),
        ("muleshoe", "test"),
    )
    for building, _ in specs:
        for index in range(12):
            rgb_dir = root / building / "rgb"
            mask_dir = root / building / "segment_semantic"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (image_size, image_size), (30, 40, 50))
            mask = Image.new("L", (image_size, image_size), 1)
            image_draw, mask_draw = ImageDraw.Draw(image), ImageDraw.Draw(mask)
            class_id = 3 + index % 5
            offset = 2 + index % 4
            box = (offset, offset, image_size // 2 + offset, image_size // 2 + offset)
            color = (40 * class_id % 255, 70 * class_id % 255, 90 * class_id % 255)
            image_draw.rectangle(box, fill=color)
            mask_draw.rectangle(box, fill=class_id)
            stem = f"point_{index:03d}_view_0"
            image.save(rgb_dir / f"{stem}_domain_rgb.png")
            mask.save(mask_dir / f"{stem}_domain_segmentsemantic.png")
