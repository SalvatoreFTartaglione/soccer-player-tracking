"""Visualisation: render annotated videos from tracking files and compare GT vs predictions interactively."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from soccer_tracker.behavior import load_rois, roi_pixel_rect
from soccer_tracker.io_utils import (
    list_frames,
    list_sequences,
    read_tracking_file,
    tracking_filename,
)

logger = logging.getLogger("soccer_tracker")

ROI_COLORS = {1: (255, 0, 0), 2: (0, 0, 255)}  # BGR: region 1 blue, region 2 red
OUTSIDE_COLOR = (0, 255, 255)  # yellow


def _id_color(track_id: int) -> Tuple[int, int, int]:
    """Stable, well separated colour per identity."""
    rng = np.random.RandomState(int(track_id) * 7919 % 2**31)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


def draw_frame(
    img: np.ndarray,
    boxes: pd.DataFrame,
    rois: Optional[Dict[int, dict]] = None,
    color_by_roi: bool = False,
    title: str = "",
) -> np.ndarray:
    """Draw boxes (+ IDs, base-centre points) and optional ROI rectangles with live counts on ``img``."""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    rects = {rid: roi_pixel_rect(r, w, h) for rid, r in (rois or {}).items()}
    counts = {rid: 0 for rid in rects}

    for row in boxes.itertuples(index=False):
        x1, y1 = int(round(row.x1)), int(round(row.y1))
        x2, y2 = int(round(row.x1 + row.w)), int(round(row.y1 + row.h))
        fx, fy = (x1 + x2) // 2, y2  # base centre = point tested against the ROIs
        color = _id_color(row.track_id)
        for rid, (rx0, ry0, rx1, ry1) in rects.items():
            if rx0 <= fx <= rx1 and ry0 <= fy <= ry1:
                counts[rid] += 1
                if color_by_roi:
                    color = ROI_COLORS.get(rid, color)
                break
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (fx, fy), 3, color, -1)
        cv2.putText(out, str(int(row.track_id)), (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    for rid, (rx0, ry0, rx1, ry1) in rects.items():
        color = ROI_COLORS.get(rid, (0, 255, 0))
        cv2.rectangle(out, (int(rx0), int(ry0)), (int(rx1), int(ry1)), color, 2)
        cv2.putText(out, f"ROI {rid}: {counts[rid]}", (int(rx0) + 8, int(ry0) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_video(
    frames_dir: Union[str, Path],
    tracking_file: Union[str, Path],
    output: Union[str, Path] = "tracking_output.mp4",
    roi_json: Optional[Union[str, Path]] = None,
    fps: float = 25.0,
    max_frames: Optional[int] = None,
) -> Path:
    """Render a sequence with tracked boxes/IDs (and ROI counts) to an MP4 file. Works headless."""
    import cv2

    frames = list_frames(frames_dir)[:max_frames]
    if not frames:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    tracks = read_tracking_file(tracking_file)
    by_frame = {f: g for f, g in tracks.groupby("frame_id")}
    rois = load_rois(roi_json) if roi_json else None

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    empty = tracks.iloc[0:0]
    try:
        for i, path in enumerate(frames, start=1):
            img = cv2.imread(str(path))
            if img is None:
                continue
            writer.write(draw_frame(img, by_frame.get(i, empty), rois, color_by_roi=bool(rois), title=f"frame {i}/{len(frames)}"))
    finally:
        writer.release()
    logger.info("Video saved to %s", output.resolve())
    return output


def compare_gt_vs_predictions(
    dataset_root: Union[str, Path],
    predictions_root: Union[str, Path],
    team_id: Union[int, str] = 27,
    fps: float = 25.0,
) -> None:
    """Interactive GT (left) | prediction (right) player. Keys: space pause, n next video, q/Esc quit."""
    import cv2

    window = "GT (left) | PRED (right)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    delay = max(1, int(round(1000.0 / max(fps, 1e-6))))

    for seq in list_sequences(dataset_root):
        gt_file = seq / "gt" / "gt.txt"
        pred_file = Path(predictions_root) / tracking_filename(seq.name, team_id)
        if not gt_file.is_file() or not pred_file.is_file():
            logger.warning("Skipping %s: missing ground truth or predictions", seq.name)
            continue
        gt, pred = read_tracking_file(gt_file), read_tracking_file(pred_file)
        gt_by, pred_by = {f: g for f, g in gt.groupby("frame_id")}, {f: g for f, g in pred.groupby("frame_id")}
        roi_path = seq / "roi.json"
        rois = load_rois(roi_path) if roi_path.is_file() else None
        frames = list_frames(seq / "img1")
        empty = gt.iloc[0:0]

        i, paused = 0, False
        while i < len(frames):
            if not paused:
                img = cv2.imread(str(frames[i]))
                i += 1
                if img is None:
                    continue
                left = draw_frame(img, gt_by.get(i, empty), rois, True, f"GT | {seq.name} | frame {i}")
                right = draw_frame(img, pred_by.get(i, empty), rois, True, f"PRED | {seq.name} | frame {i}")
                cv2.imshow(window, cv2.hconcat([left, right]))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = not paused
            if key == ord("n"):
                break
    cv2.destroyAllWindows()
