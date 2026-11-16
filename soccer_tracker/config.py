"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at top level")
    return data


def resolve_device(requested: str | int | None = "auto") -> str:
    """Map ``auto``/``cpu``/``0``/``cuda:0`` to a device string understood by Ultralytics and BoxMOT."""
    if requested in (None, "auto"):
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return str(requested)
