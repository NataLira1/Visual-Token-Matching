from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def environment_info() -> dict[str, str]:
    try:
        import timm
        timm_version = timm.__version__
    except ImportError:
        timm_version = "not-installed"
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "timm": timm_version,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
