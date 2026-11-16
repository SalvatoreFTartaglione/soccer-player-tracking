"""Offline post-processing of tracker output.

Online trackers fragment identities after long occlusions and produce jittery boxes. This module
refines a finished sequence in four steps:

1. :func:`stitch_tracklets`       - re-link fragmented tracks with a constant-velocity motion model,
2. :func:`interpolate_and_smooth` - fill short detection gaps and low-pass filter the boxes,
3. :func:`remove_short_tracks`    - drop transient "ghost" tracks,
4. :func:`relabel_ids`            - renumber identities ``1..N`` as required by the contest format.

All functions take and return DataFrames with the columns of :data:`soccer_tracker.io_utils.TRACK_COLUMNS`
(``frame_id, track_id, x1, y1, w, h``) and never modify their input.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

from soccer_tracker.io_utils import TRACK_COLUMNS, read_tracking_file, write_tracking_file

logger = logging.getLogger("soccer_tracker")

_INFEASIBLE = 1e9


@dataclass
class PostprocessConfig:
    """Post-processing hyper-parameters (frames at 25 fps, pixels at 1920x1080)."""

    stitch: bool = True
    stitch_max_time_gap: int = 50        # max frames between the end of a tracklet and the start of the next
    stitch_max_space_dist: float = 150.0  # max distance (px) between predicted and observed position
    stitch_max_speed: float = 20.0       # max implied speed (px/frame) of the bridging motion
    stitch_velocity_window: int = 5      # frames used to estimate the exit velocity of a tracklet
    interpolate: bool = True
    interp_max_gap: int = 10             # gaps longer than this many missing frames are left empty
    smooth_sigma: float = 2.0            # Gaussian smoothing (frames); 0 disables smoothing
    min_track_len: int = 50              # tracks shorter than this many frames are removed (2 s at 25 fps)

    @classmethod
    def from_dict(cls, data: Optional[Mapping]) -> "PostprocessConfig":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown postprocess options: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> Dict:
        return asdict(self)


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=df[c].dtype if c in df else float) for c in TRACK_COLUMNS})


def _tracklet_summaries(df: pd.DataFrame, velocity_window: int) -> Dict[int, dict]:
    """Start/end frame, start/end centre and exit velocity (px/frame) for every track."""
    summaries: Dict[int, dict] = {}
    ordered = df.sort_values(["track_id", "frame_id"])
    for tid, g in ordered.groupby("track_id", sort=False):
        frames = g["frame_id"].to_numpy()
        cx = (g["x1"] + g["w"] / 2.0).to_numpy()
        cy = (g["y1"] + g["h"] / 2.0).to_numpy()
        n = min(velocity_window, len(frames))
        dt = frames[-1] - frames[-n]
        if n > 1 and dt > 0:
            velocity = np.array([(cx[-1] - cx[-n]) / dt, (cy[-1] - cy[-n]) / dt])
        else:
            velocity = np.zeros(2)
        summaries[int(tid)] = {
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "start_pos": np.array([cx[0], cy[0]]),
            "end_pos": np.array([cx[-1], cy[-1]]),
            "velocity": velocity,
        }
    return summaries


def stitch_tracklets(
    df: pd.DataFrame,
    max_time_gap: int = 50,
    max_space_dist: float = 150.0,
    max_speed: float = 20.0,
    velocity_window: int = 5,
) -> pd.DataFrame:
    """Merge track fragments that plausibly belong to the same player.

    A tracklet ``a`` may be followed by tracklet ``b`` when ``b`` starts after ``a`` ends, within
    ``max_time_gap`` frames, and the constant-velocity extrapolation of ``a`` (from the box centre,
    using its last ``velocity_window`` frames) lands within ``max_space_dist`` pixels of ``b``'s first
    box, with an implied bridging speed below ``max_speed`` px/frame. Candidate pairs are resolved with
    a global minimum-cost one-to-one assignment (Hungarian algorithm) on the extrapolation error, so a
    tracklet gets at most one successor and one predecessor. Because the constraints only involve the
    original tracklet end-points, chains ``a -> b -> c`` of any length are merged correctly.

    Returns a copy of ``df`` in which all tracklets of a chain share the track id of its first element.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    df["frame_id"] = df["frame_id"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    summaries = _tracklet_summaries(df, velocity_window)
    ids = list(summaries)
    n = len(ids)
    if n < 2:
        return df

    cost = np.full((n, n), _INFEASIBLE)
    for i, a in enumerate(ids):
        sa = summaries[a]
        for j, b in enumerate(ids):
            if i == j:
                continue
            sb = summaries[b]
            gap = sb["start_frame"] - sa["end_frame"]
            if gap <= 0 or gap > max_time_gap:
                continue
            predicted = sa["end_pos"] + sa["velocity"] * gap
            error = float(np.linalg.norm(predicted - sb["start_pos"]))
            speed = float(np.linalg.norm(sb["start_pos"] - sa["end_pos"])) / gap
            if error < max_space_dist and speed < max_speed:
                cost[i, j] = error

    rows, cols = linear_sum_assignment(cost)
    successor = {ids[i]: ids[j] for i, j in zip(rows, cols) if cost[i, j] < _INFEASIBLE}
    has_predecessor = set(successor.values())

    id_map: Dict[int, int] = {}
    for root in ids:
        if root in has_predecessor:
            continue
        node = root
        while True:
            id_map[node] = root
            if node not in successor:
                break
            node = successor[node]

    n_merged = n - len(set(id_map.values()))
    logger.debug("stitch_tracklets: %d tracklets -> %d identities (%d merges)", n, n - n_merged, n_merged)
    df["track_id"] = df["track_id"].map(id_map).astype(int)
    return df


def interpolate_and_smooth(df: pd.DataFrame, max_gap: int = 10, sigma: float = 2.0) -> pd.DataFrame:
    """Linearly interpolate short gaps and Gaussian-smooth the box coordinates of every track.

    Gaps of up to ``max_gap`` missing frames are filled by linear interpolation of ``x1, y1, w, h``.
    Longer gaps are left empty (no motion is hallucinated) and split the track into independent
    segments; smoothing (``sigma`` in frames, 0 to disable) is applied within each segment only, so
    boxes on either side of a gap never influence each other.
    """
    if df is None or df.empty:
        return df
    df = df.sort_values(["track_id", "frame_id"]).drop_duplicates(["track_id", "frame_id"])

    pieces = []
    for tid, g in df.groupby("track_id", sort=False):
        frames = g["frame_id"].to_numpy(dtype=int)
        values = g[["x1", "y1", "w", "h"]].to_numpy(dtype=float)
        missing = np.diff(frames) - 1
        breaks = np.where(missing > max_gap)[0] + 1
        for seg_frames, seg_values in zip(np.split(frames, breaks), np.split(values, breaks)):
            full = np.arange(seg_frames[0], seg_frames[-1] + 1)
            filled = np.column_stack([np.interp(full, seg_frames, seg_values[:, k]) for k in range(4)])
            if sigma > 0 and len(full) > 1:
                filled = gaussian_filter1d(filled, sigma=sigma, axis=0, mode="nearest")
            pieces.append(
                pd.DataFrame(
                    {
                        "frame_id": full,
                        "track_id": int(tid),
                        "x1": filled[:, 0],
                        "y1": filled[:, 1],
                        "w": filled[:, 2],
                        "h": filled[:, 3],
                    }
                )
            )
    if not pieces:
        return _empty_like(df)
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["frame_id", "track_id"]).reset_index(drop=True)


def remove_short_tracks(df: pd.DataFrame, min_len: int = 50) -> pd.DataFrame:
    """Drop tracks that appear in fewer than ``min_len`` frames (ghost tracks / crowd detections)."""
    if df is None or df.empty:
        return df
    counts = df["track_id"].value_counts()
    keep = counts[counts >= min_len].index
    return df[df["track_id"].isin(keep)].copy()


def relabel_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber track ids ``1..N`` in order of first appearance (contest format starts at 1)."""
    if df is None or df.empty:
        return df
    df = df.copy()
    first_seen = df.groupby("track_id")["frame_id"].min().sort_values(kind="stable")
    mapping = {old: new for new, old in enumerate(first_seen.index, start=1)}
    df["track_id"] = df["track_id"].map(mapping).astype(int)
    return df


def postprocess_dataframe(df: pd.DataFrame, cfg: Optional[PostprocessConfig] = None) -> pd.DataFrame:
    """Run the full post-processing chain on one sequence."""
    cfg = cfg or PostprocessConfig()
    if df is None or df.empty:
        return df
    if cfg.stitch:
        df = stitch_tracklets(
            df,
            max_time_gap=cfg.stitch_max_time_gap,
            max_space_dist=cfg.stitch_max_space_dist,
            max_speed=cfg.stitch_max_speed,
            velocity_window=cfg.stitch_velocity_window,
        )
    if cfg.interpolate:
        df = interpolate_and_smooth(df, max_gap=cfg.interp_max_gap, sigma=cfg.smooth_sigma)
    if cfg.min_track_len > 0:
        df = remove_short_tracks(df, min_len=cfg.min_track_len)
    return relabel_ids(df)


def postprocess_folder(folder: Path | str, cfg: Optional[PostprocessConfig] = None) -> None:
    """Post-process every ``tracking_*.txt`` file in ``folder`` in place."""
    folder = Path(folder)
    for txt in sorted(folder.glob("tracking_*.txt")):
        df = read_tracking_file(txt)
        if df.empty:
            continue
        write_tracking_file(postprocess_dataframe(df, cfg), txt)
        logger.info("Post-processed %s", txt.name)
