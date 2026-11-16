"""Contest metrics: HOTA@0.5 (TrackEval, with DetA/AssA) for tracking and nMAE for behaviour analysis.

Expected ground-truth layout (one folder per sequence)::

    <dataset_root>/<seq>/img1/*.jpg
    <dataset_root>/<seq>/gt/gt.txt              MOT format: frame,id,x,y,w,h,...
    <dataset_root>/<seq>/gt/behavior_gt.txt     optional, frame,region_id,n_players
    <dataset_root>/<seq>/gameinfo.ini           optional (SoccerNet): used to drop the ball from the GT

Predictions: ``<predictions_root>/tracking_<seq>_<team>.txt`` and ``behavior_<seq>_<team>.txt``.
The final contest score is ``PTBS = HOTA@0.5 + nMAE``.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from soccer_tracker.io_utils import (
    behavior_filename,
    list_frames,
    list_sequences,
    read_gameinfo_labels,
    tracking_filename,
)

logger = logging.getLogger("soccer_tracker")

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- behaviour metric
def read_behavior_file(path: PathLike) -> Dict[Tuple[int, int], int]:
    """``{(frame, region_id): n_players}`` from a behaviour file."""
    out: Dict[Tuple[int, int], int] = {}
    with Path(path).open("r", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 3:
                out[(int(row[0]), int(row[1]))] = int(float(row[2]))
    return out


def compute_nmae(
    dataset_root: PathLike, predictions_root: PathLike, team_id: Union[int, str], only: Optional[Iterable[str]] = None
) -> Dict[str, Optional[float]]:
    """Mean absolute error of the ROI player counts over all sequences, and its normalised score.

    ``nMAE = (10 - min(10, MAE)) / 10`` (1 is perfect). The error is computed on every entry of the
    ground truth; a missing prediction entry counts as 0 players. Returns ``None`` values when some
    sequence lacks a ground-truth or prediction behaviour file.
    """
    abs_err, n, complete = 0.0, 0, True
    for seq in list_sequences(dataset_root, only):
        gt_path = seq / "gt" / "behavior_gt.txt"
        pr_path = Path(predictions_root) / behavior_filename(seq.name, team_id)
        if not (gt_path.is_file() and pr_path.is_file()):
            complete = False
            continue
        gt, pred = read_behavior_file(gt_path), read_behavior_file(pr_path)
        for key, gt_val in gt.items():
            abs_err += abs(pred.get(key, 0) - gt_val)
            n += 1
    if not complete or n == 0:
        return {"MAE": None, "nMAE": None}
    mae = abs_err / n
    return {"MAE": mae, "nMAE": (10.0 - min(10.0, mae)) / 10.0}


# --------------------------------------------------------------------------- tracking metric
def _patch_numpy_aliases() -> None:
    """TrackEval still uses ``np.float`` & co., removed in NumPy 1.24."""
    for alias, target in (("float", float), ("int", int), ("bool", bool)):
        if vars(np).get(alias) is None:  # (hasattr would trigger NumPy's deprecation FutureWarning)
            setattr(np, alias, target)


def _write_mot_10col(src: Path, dst: Path, drop_ids: frozenset = frozenset()) -> None:
    """Rewrite a MOT file with 10 columns, forcing class 1 (pedestrian) and dropping ``drop_ids``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    with src.open("r") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 6:
                continue
            try:
                if int(float(parts[1])) in drop_ids:
                    continue
            except ValueError:
                continue
            conf = parts[6] if len(parts) >= 7 else "1"
            vis = parts[8] if len(parts) >= 9 else "-1"
            z = parts[9] if len(parts) >= 10 else "-1"
            lines.append(",".join([parts[0], parts[1], *parts[2:6], conf, "1", vis, z]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def _build_trackeval_tree(
    dataset_root: PathLike,
    predictions_root: PathLike,
    team_id: Union[int, str],
    split: str,
    fps: float,
    tmp_root: Path,
    benchmark: str,
    tracker_name: str,
    exclude_ball: bool,
    only: Optional[Iterable[str]] = None,
) -> Tuple[str, str, str]:
    """Lay out ground truth and predictions in the MOTChallenge2DBox structure TrackEval expects."""
    import cv2

    gt_root = tmp_root / "gt" / f"{benchmark}-{split}"
    pred_root = tmp_root / "trackers" / f"{benchmark}-{split}" / tracker_name / "data"
    seqmap_dir = tmp_root / "seqmaps"
    for d in (gt_root, pred_root, seqmap_dir):
        d.mkdir(parents=True, exist_ok=True)

    sequences = list_sequences(dataset_root, only)
    if not sequences:
        raise FileNotFoundError(f"No sequences with an 'img1' folder found in {dataset_root}")

    for seq in sequences:
        gt_src = seq / "gt" / "gt.txt"
        pred_src = Path(predictions_root) / tracking_filename(seq.name, team_id)
        if not gt_src.is_file():
            raise FileNotFoundError(f"Missing ground truth: {gt_src}")
        if not pred_src.is_file():
            raise FileNotFoundError(f"Missing prediction file: {pred_src}")

        frames = list_frames(seq / "img1")
        first = cv2.imread(str(frames[0])) if frames else None
        if first is None:
            raise RuntimeError(f"Cannot read frames of {seq}")
        height, width = first.shape[:2]

        ball_ids = frozenset()
        if exclude_ball:
            ball_ids = frozenset(i for i, lbl in read_gameinfo_labels(seq / "gameinfo.ini").items() if "ball" in lbl)

        seq_dir = gt_root / seq.name
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        (seq_dir / "img1").mkdir(exist_ok=True)
        (seq_dir / "seqinfo.ini").write_text(
            "\n".join(
                [
                    "[Sequence]",
                    f"name={seq.name}",
                    "imDir=img1",
                    f"frameRate={int(round(fps))}",
                    f"seqLength={len(frames)}",
                    f"imWidth={width}",
                    f"imHeight={height}",
                    "imExt=.jpg",
                    "",
                ]
            )
        )
        _write_mot_10col(gt_src, seq_dir / "gt" / "gt.txt", drop_ids=ball_ids)
        _write_mot_10col(pred_src, pred_root / f"{seq.name}.txt")

    seqmap = seqmap_dir / f"{benchmark}-{split}.txt"
    seqmap.write_text("name\n" + "\n".join(s.name for s in sequences) + "\n")
    return str(tmp_root / "gt"), str(tmp_root / "trackers"), str(seqmap)


def compute_hota05(
    dataset_root: PathLike,
    predictions_root: PathLike,
    team_id: Union[int, str],
    fps: float = 25.0,
    split: str = "test",
    exclude_ball: bool = True,
    verbose: bool = False,
    only: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """HOTA, DetA and AssA at IoU threshold 0.5 over all sequences (TrackEval, combined)."""
    _patch_numpy_aliases()
    import trackeval

    benchmark, tracker_name = "SNMOT", "test"
    with tempfile.TemporaryDirectory(prefix="soccer_trackeval_") as tmp:
        gt_folder, tr_folder, seqmap = _build_trackeval_tree(
            dataset_root, predictions_root, team_id, split, fps, Path(tmp), benchmark, tracker_name, exclude_ball, only
        )
        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config.update(
            {
                "DISPLAY_LESS_PROGRESS": True,
                "PRINT_RESULTS": verbose,
                "PRINT_CONFIG": False,
                "TIME_PROGRESS": False,
                "OUTPUT_SUMMARY": False,
                "OUTPUT_EMPTY_CLASSES": False,
                "OUTPUT_DETAILED": False,
                "PLOT_CURVES": False,
                "LOG_ON_ERROR": os.path.join(tmp, "error_log.txt"),
            }
        )
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update(
            {
                "BENCHMARK": benchmark,
                "GT_FOLDER": gt_folder,
                "TRACKERS_FOLDER": tr_folder,
                "TRACKERS_TO_EVAL": [tracker_name],
                "SPLIT_TO_EVAL": split,
                "SEQMAP_FILE": seqmap,
                "DO_PREPROC": False,
                "PRINT_CONFIG": False,
                "TRACKER_SUB_FOLDER": "data",
                "OUTPUT_SUB_FOLDER": "eval_results",
            }
        )
        metric = trackeval.metrics.HOTA({"METRICS": ["HOTA"], "PRINT_CONFIG": False})
        evaluator = trackeval.Evaluator(eval_config)
        output, _ = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(dataset_config)], [metric])

    idx = int(np.where(np.isclose(np.array(metric.array_labels, dtype=float), 0.5))[0][0])
    combined = output["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]["HOTA"]
    return {name: float(combined[name][idx]) for name in ("HOTA", "DetA", "AssA")}


def evaluate(
    dataset_root: PathLike,
    predictions_root: PathLike,
    team_id: Union[int, str] = 27,
    fps: float = 25.0,
    split: str = "test",
    exclude_ball: bool = True,
    only: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[float]]:
    """Full contest evaluation: HOTA@0.5 (+DetA, AssA), MAE, nMAE and PTBS = HOTA@0.5 + nMAE.

    When behaviour ground truth/predictions are missing, ``nMAE`` is ``None`` and ``PTBS`` equals HOTA@0.5.
    """
    result: Dict[str, Optional[float]] = dict(
        compute_hota05(dataset_root, predictions_root, team_id, fps, split, exclude_ball, only=only)
    )
    result.update(compute_nmae(dataset_root, predictions_root, team_id, only))
    result["PTBS"] = result["HOTA"] + (result["nMAE"] or 0.0)
    return result


def format_report(result: Dict[str, Optional[float]]) -> str:
    def f(key: str) -> str:
        v = result.get(key)
        return "n/a" if v is None else f"{v:.4f}"

    return "\n".join(
        [
            "=" * 40,
            f"HOTA@0.5 : {f('HOTA')}   (DetA {f('DetA')}, AssA {f('AssA')})",
            f"MAE      : {f('MAE')}",
            f"nMAE     : {f('nMAE')}",
            f"PTBS     : {f('PTBS')}",
            "=" * 40,
        ]
    )
