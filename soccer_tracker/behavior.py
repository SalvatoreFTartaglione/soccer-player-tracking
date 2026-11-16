"""Player behaviour analysis: how many players stand inside each region of interest (ROI).

Each test video comes with a ``roi.json`` holding two rectangles in normalised coordinates::

    {"roi1": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}, "roi2": {...}}

A player counts as being inside an ROI when the *centre of the base of its bounding box* (the point
between the feet) lies inside the rectangle (borders included).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from soccer_tracker.io_utils import (
    behavior_filename,
    list_frames,
    list_sequences,
    read_tracking_file,
    tracking_filename,
)

logger = logging.getLogger("soccer_tracker")

DEFAULT_RESOLUTION = (1920, 1080)
Roi = Dict[str, float]


def load_rois(roi_json: Union[str, Path]) -> Dict[int, Roi]:
    """Load ``roi.json`` and return ``{region_id: roi}`` with ids taken from the key suffix (``roi2`` -> 2)."""
    with Path(roi_json).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    rois: Dict[int, Roi] = {}
    for position, key in enumerate(sorted(raw), start=1):
        m = re.search(r"(\d+)$", key)
        rois[int(m.group(1)) if m else position] = raw[key]
    return rois


def roi_pixel_rect(roi: Roi, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Denormalise an ROI into ``(x_min, y_min, x_max, y_max)`` pixel coordinates."""
    x = roi["x"] * img_w
    y = roi["y"] * img_h
    return x, y, x + roi["width"] * img_w, y + roi["height"] * img_h


def feet_points(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Base-centre of every box: ``(x + w/2, y + h)``."""
    return (df["x1"] + df["w"] / 2.0).to_numpy(), (df["y1"] + df["h"]).to_numpy()


def count_players_in_rois(
    tracks: pd.DataFrame,
    rois: Dict[int, Roi],
    img_w: int = DEFAULT_RESOLUTION[0],
    img_h: int = DEFAULT_RESOLUTION[1],
    n_frames: Optional[int] = None,
) -> pd.DataFrame:
    """Per-frame player count in each ROI.

    Returns a DataFrame ``frame_id, region_id, n_players`` that contains *every* frame in
    ``1..n_frames`` and every region (frames with nobody inside are reported as 0, not omitted).
    ``n_frames`` defaults to the last frame that has a track.
    """
    last_tracked = int(tracks["frame_id"].max()) if len(tracks) else 0
    n_frames = max(int(n_frames or 0), last_tracked)
    all_frames = np.arange(1, n_frames + 1)

    px, py = feet_points(tracks) if len(tracks) else (np.empty(0), np.empty(0))
    blocks = []
    for region_id in sorted(rois):
        x0, y0, x1, y1 = roi_pixel_rect(rois[region_id], img_w, img_h)
        inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        per_frame = tracks.loc[inside].groupby("frame_id").size() if len(tracks) else pd.Series(dtype=int)
        counts = per_frame.reindex(all_frames, fill_value=0).astype(int)
        blocks.append(
            pd.DataFrame({"frame_id": all_frames, "region_id": region_id, "n_players": counts.to_numpy()})
        )
    if not blocks:
        return pd.DataFrame(columns=["frame_id", "region_id", "n_players"])
    return pd.concat(blocks, ignore_index=True).sort_values(["frame_id", "region_id"]).reset_index(drop=True)


def write_behavior_file(counts: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write ``frame_id,region_id,n_players`` lines (contest format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts[["frame_id", "region_id", "n_players"]].to_csv(path, index=False, header=False)


def _image_size(frames_dir: Path) -> Tuple[int, int]:
    """(width, height) of the first frame, falling back to the contest resolution."""
    try:
        import cv2

        frames = list_frames(frames_dir)
        img = cv2.imread(str(frames[0])) if frames else None
        if img is not None:
            return img.shape[1], img.shape[0]
    except Exception:  # pragma: no cover - best effort only
        pass
    return DEFAULT_RESOLUTION


def generate_behavior_analysis(
    tracking_folder: Union[str, Path],
    sequences_root: Union[str, Path],
    team_id: Union[int, str] = 27,
    only: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Create ``behavior_<seq>_<team>.txt`` next to every tracking file of ``tracking_folder``.

    For each sequence under ``sequences_root`` the function needs ``<seq>/roi.json``; sequences without
    one are skipped with a warning. Image size and frame count are read from ``<seq>/img1``.
    """
    tracking_folder = Path(tracking_folder)
    written = []
    for seq_dir in list_sequences(sequences_root, only):
        track_path = tracking_folder / tracking_filename(seq_dir.name, team_id)
        roi_path = seq_dir / "roi.json"
        if not track_path.is_file():
            logger.warning("No tracking file for %s (%s) - skipped", seq_dir.name, track_path.name)
            continue
        if not roi_path.is_file():
            logger.warning("No roi.json for %s - behaviour analysis skipped", seq_dir.name)
            continue
        tracks = read_tracking_file(track_path)
        img_w, img_h = _image_size(seq_dir / "img1")
        n_frames = len(list_frames(seq_dir / "img1"))
        counts = count_players_in_rois(tracks, load_rois(roi_path), img_w, img_h, n_frames)
        out = tracking_folder / behavior_filename(seq_dir.name, team_id)
        write_behavior_file(counts, out)
        written.append(out)
        logger.info("Behaviour analysis written to %s", out.name)
    return written
