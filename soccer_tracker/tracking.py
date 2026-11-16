"""Tracking-by-detection: YOLO11 detections -> BoT-SORT (camera-motion compensation + fine-tuned OSNet ReID).

The detector and the tracker are decoupled: :func:`iter_detections` yields raw detections frame by
frame, :func:`run_tracker` feeds them to any object exposing ``update(dets, img)`` (BoxMOT's
``BotSort`` in production, a stub in the unit tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from soccer_tracker.config import load_yaml, resolve_device
from soccer_tracker.io_utils import (
    TRACK_COLUMNS,
    frame_number,
    list_frames,
    list_sequences,
    tracking_filename,
    write_tracking_file,
)
from soccer_tracker.postprocess import PostprocessConfig, postprocess_dataframe

logger = logging.getLogger("soccer_tracker")

#: Keys of the tracker YAML forwarded to ``boxmot.BotSort``.
BOTSORT_KEYS = (
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "track_buffer",
    "match_thresh",
    "proximity_thresh",
    "appearance_thresh",
    "frame_rate",
    "with_reid",
    "cmc_method",
    "fuse_first_associate",
    # accepted by BoxMOT's BaseTracker (kwargs); documented as having no effect in boxmot 16.0.1
    "det_thresh",
    "iou_threshold",
)


# --------------------------------------------------------------------------- detection
def load_detector(weights: Union[str, Path]):
    """Load a trained Ultralytics YOLO detector."""
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights}")
    return YOLO(str(weights))


def iter_detections(
    detector,
    frames: List[Path],
    imgsz: int = 1920,
    conf: float = 0.20,
    iou: float = 0.65,
    device: str = "auto",
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Run the detector over ``frames`` (already sorted) and yield ``(frame_id, dets, image)``.

    ``dets`` is an ``(N, 6)`` float array ``[x1, y1, x2, y2, conf, cls]`` restricted to class 0
    (``Person``: players, goalkeepers and referees).

    Frames are processed one at a time, in natural numeric order. (Do **not** pass the whole list to
    ``predict``: Ultralytics would run it as a single batch and exhaust GPU/host memory, and passing a
    folder relies on lexicographic order, which breaks for non zero-padded file names.)
    """
    device = resolve_device(device)
    for position, path in enumerate(frames, start=1):
        result = detector.predict(
            source=str(path), imgsz=imgsz, conf=conf, iou=iou, classes=[0], verbose=False, device=device
        )[0]
        yield frame_number(path, fallback=position), result.boxes.data.cpu().numpy().astype(np.float32), result.orig_img


# --------------------------------------------------------------------------- tracker
def load_tracker_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Read the BoT-SORT YAML and keep only the keys forwarded to BoxMOT (unknown keys are reported)."""
    raw = load_yaml(path)
    unknown = set(raw) - set(BOTSORT_KEYS) - {"tracker_type"}
    if unknown:
        logger.warning("Ignoring unknown tracker options: %s", sorted(unknown))
    return {k: raw[k] for k in BOTSORT_KEYS if k in raw}


def build_tracker(tracker_cfg: Dict[str, Any], reid_weights: Union[str, Path], device: str = "auto", half: bool = False):
    """Instantiate a fresh BoxMOT BoT-SORT tracker (one per video, so no state leaks across sequences)."""
    from boxmot import BotSort

    reid_weights = Path(reid_weights)
    if tracker_cfg.get("with_reid", True) and not reid_weights.is_file():
        raise FileNotFoundError(f"ReID weights not found: {reid_weights}")
    logging.getLogger("boxmot").setLevel(logging.WARNING)
    return BotSort(reid_weights=reid_weights, device=resolve_device(device), half=half, **tracker_cfg)


def run_tracker(detections: Iterable[Tuple[int, np.ndarray, Optional[np.ndarray]]], tracker) -> pd.DataFrame:
    """Feed per-frame detections to ``tracker`` and collect the tracks as a DataFrame.

    The tracker is updated on **every** frame, including frames without detections, so that its
    internal clock advances and lost tracks age (and expire) correctly.
    """
    rows = []
    for frame_id, dets, image in detections:
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
        tracks = np.asarray(tracker.update(dets, image)).reshape(-1, 8)
        for x1, y1, x2, y2, track_id, *_ in tracks:
            rows.append((frame_id, int(track_id), float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    return df.astype({"frame_id": int, "track_id": int})


# --------------------------------------------------------------------------- orchestration
@dataclass
class TrackingConfig:
    """Everything the tracking stage needs (see ``configs/pipeline.yaml``)."""

    detector_weights: str = "weights/yolo11l_soccer.pt"
    reid_weights: str = "weights/osnet_ain_x1_0_soccer.pt"
    tracker_config: str = "configs/botsort_soccer.yaml"
    imgsz: int = 1920
    conf: float = 0.20
    iou: float = 0.65
    device: str = "auto"
    team_id: int = 27
    save_raw: bool = True
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    @classmethod
    def from_pipeline_yaml(cls, path: Union[str, Path]) -> "TrackingConfig":
        y = load_yaml(path)
        det, reid, trk = y.get("detector", {}), y.get("reid", {}), y.get("tracker", {})
        return cls(
            detector_weights=det.get("weights", cls.detector_weights),
            reid_weights=reid.get("weights", cls.reid_weights),
            tracker_config=trk.get("config", cls.tracker_config),
            imgsz=det.get("imgsz", cls.imgsz),
            conf=det.get("conf", cls.conf),
            iou=det.get("iou", cls.iou),
            device=str(y.get("device", cls.device)),
            team_id=int(y.get("team_id", cls.team_id)),
            save_raw=bool(y.get("save_raw", cls.save_raw)),
            postprocess=PostprocessConfig.from_dict(y.get("postprocess")),
        )


def track_sequences(
    sequences_root: Union[str, Path],
    output_dir: Union[str, Path],
    cfg: TrackingConfig,
    postprocess: bool = True,
    only: Optional[Iterable[str]] = None,
) -> List[Path]:
    """Detect, track and (optionally) post-process every sequence under ``sequences_root``.

    Writes ``tracking_<seq>_<team>.txt`` into ``output_dir`` (and the un-processed tracker output to
    ``output_dir/raw`` when ``cfg.save_raw`` is set). ``only`` restricts the run to the named sequences.
    Returns the list of final files.
    """
    from time import time

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = list_sequences(sequences_root, only)
    if not sequences:
        raise FileNotFoundError(f"No sequences (folders with an 'img1' directory) found in {sequences_root}")

    detector = load_detector(cfg.detector_weights)
    tracker_cfg = load_tracker_config(cfg.tracker_config)
    logger.info("Tracker configuration: %s", tracker_cfg)

    written = []
    t0 = time()
    for i, seq in enumerate(sequences, start=1):
        fname = tracking_filename(seq.name, cfg.team_id)
        frames = list_frames(seq / "img1")
        logger.info("[%d/%d] %s: %d frames", i, len(sequences), seq.name, len(frames))
        tracker = build_tracker(tracker_cfg, cfg.reid_weights, cfg.device)
        dets = iter_detections(detector, frames, cfg.imgsz, cfg.conf, cfg.iou, cfg.device)
        df = run_tracker(dets, tracker)
        if cfg.save_raw:
            write_tracking_file(df, output_dir / "raw" / fname)
        if postprocess:
            df = postprocess_dataframe(df, cfg.postprocess)
        write_tracking_file(df, output_dir / fname)
        written.append(output_dir / fname)
    logger.info("Tracking done in %.1f min", (time() - t0) / 60)
    return written
