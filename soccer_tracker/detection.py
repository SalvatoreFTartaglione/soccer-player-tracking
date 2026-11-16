"""Detector fine-tuning, validation and training-curve plotting (Ultralytics YOLO11)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from soccer_tracker.config import load_yaml, resolve_device

logger = logging.getLogger("soccer_tracker")


def train_detector(
    data_yaml: Union[str, Path],
    config: Union[str, Path] = "configs/yolo_train.yaml",
    project: Union[str, Path] = "runs/detect",
    name: str = "soccer_tracker",
    overrides: Optional[Dict[str, Any]] = None,
    resume: bool = False,
    device: str = "auto",
) -> Path:
    """Fine-tune YOLO11 on the converted SoccerNet dataset and return the path of ``best.pt``.

    Hyper-parameters come from ``config`` (default: the exact setup of the released run, see
    ``results/detector_training/args.yaml``); ``overrides`` takes precedence. With ``resume=True`` the
    ``last.pt`` of ``project/name`` is resumed instead.
    """
    from ultralytics import YOLO

    if resume:
        last = Path(project) / name / "weights" / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(f"Nothing to resume: {last} does not exist")
        results = YOLO(str(last)).train(resume=True)
    else:
        params = load_yaml(config)
        base_model = params.pop("model", "yolo11l.pt")
        params.update(overrides or {})
        results = YOLO(base_model).train(
            data=str(data_yaml), project=str(project), name=name, device=resolve_device(device), **params
        )
    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training finished. Best weights: %s", best)
    return best


def validate_detector(
    weights: Union[str, Path],
    data_yaml: Union[str, Path],
    imgsz: int = 1920,
    conf: float = 0.001,
    iou: float = 0.65,
    split: str = "val",
    device: str = "auto",
) -> Dict[str, float]:
    """Compute precision / recall / mAP of a detector on a YOLO dataset split."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), imgsz=imgsz, conf=conf, iou=iou, split=split, device=resolve_device(device), plots=False
    )
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }


def load_results_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def best_epoch(df: pd.DataFrame) -> pd.Series:
    """Row with the highest Ultralytics fitness (0.1 * mAP50 + 0.9 * mAP50-95), i.e. the one saved as ``best.pt``."""
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    return df.loc[fitness.idxmax()]


def plot_training_metrics(csv_path: Union[str, Path], out_png: Union[str, Path], show: bool = False) -> Path:
    """Plot losses, mAP and precision/recall from an Ultralytics ``results.csv`` and save a PNG."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_results_csv(csv_path)
    epochs = df["epoch"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    panels = [
        (axs[0, 0], "Box loss (localisation)", [("train/box_loss", "train"), ("val/box_loss", "val")]),
        (axs[0, 1], "Class loss", [("train/cls_loss", "train"), ("val/cls_loss", "val")]),
        (axs[1, 0], "mAP (higher is better)", [("metrics/mAP50(B)", "mAP@50"), ("metrics/mAP50-95(B)", "mAP@50-95")]),
        (axs[1, 1], "Precision / recall", [("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")]),
    ]
    for ax, title, series in panels:
        for col, label in series:
            ax.plot(epochs, df[col], marker="o", ms=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_png
