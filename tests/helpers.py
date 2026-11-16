"""Synthetic data builders shared by the tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from soccer_tracker.io_utils import TRACK_COLUMNS


def make_track(tid, f0, f1, x0=100.0, y0=300.0, vx=2.0, vy=0.0, w=50.0, h=150.0, skip=()):
    """A player walking at constant velocity from frame f0 to f1 (inclusive), minus ``skip`` frames."""
    rows = []
    for f in range(f0, f1 + 1):
        if f in skip:
            continue
        t = f - f0
        rows.append((f, tid, x0 + vx * t, y0 + vy * t, w, h))
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def concat(*dfs):
    return pd.concat(dfs, ignore_index=True)


def make_sequence(root: Path, name: str, n_frames: int = 6, size=(320, 180), gt_rows=None, behavior_rows=None, roi=None, ini=None):
    """Create ``<root>/<name>/{img1,gt}`` with gray frames and optional GT / ROI / gameinfo files."""
    import cv2

    seq = Path(root) / name
    (seq / "img1").mkdir(parents=True, exist_ok=True)
    (seq / "gt").mkdir(exist_ok=True)
    for i in range(1, n_frames + 1):
        img = np.full((size[1], size[0], 3), 90, dtype=np.uint8)
        cv2.imwrite(str(seq / "img1" / f"{i:06d}.jpg"), img)
    if gt_rows is not None:
        pd.DataFrame(gt_rows).to_csv(seq / "gt" / "gt.txt", header=False, index=False)
    if behavior_rows is not None:
        pd.DataFrame(behavior_rows).to_csv(seq / "gt" / "behavior_gt.txt", header=False, index=False)
    if roi is not None:
        import json

        (seq / "roi.json").write_text(json.dumps(roi))
    if ini is not None:
        (seq / "gameinfo.ini").write_text(ini)
    return seq
