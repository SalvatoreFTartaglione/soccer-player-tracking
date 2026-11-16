"""Shared helpers: sequence discovery, frame listing, MOT-style text I/O, SoccerNet metadata."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

logger = logging.getLogger("soccer_tracker")

PathLike = Union[str, Path]

#: Columns of the tracking result files. ``x1, y1`` is the top-left corner of the box.
TRACK_COLUMNS = ["frame_id", "track_id", "x1", "y1", "w", "h"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- naming
def team_tag(team_id: Union[int, str]) -> str:
    """Two-digit team identifier used in the contest file names (``27`` -> ``"27"``, ``1`` -> ``"01"``)."""
    return f"{int(team_id):02d}"


def tracking_filename(seq_name: str, team_id: Union[int, str]) -> str:
    return f"tracking_{seq_name}_{team_tag(team_id)}.txt"


def behavior_filename(seq_name: str, team_id: Union[int, str]) -> str:
    return f"behavior_{seq_name}_{team_tag(team_id)}.txt"


def natural_key(path: PathLike) -> list:
    """Sort key that orders ``2.jpg`` before ``10.jpg``."""
    name = Path(path).name
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def frame_number(path: PathLike, fallback: int | None = None) -> int:
    """Frame index encoded in a frame file name (``000123.jpg`` -> ``123``)."""
    stem = Path(path).stem
    digits = re.findall(r"\d+", stem)
    if digits:
        return int(digits[-1])
    if fallback is None:
        raise ValueError(f"Cannot infer frame number from '{path}'")
    return fallback


# --------------------------------------------------------------------------- sequences
def list_frames(img_dir: PathLike) -> List[Path]:
    """All image files of a sequence, in natural (numeric) order."""
    img_dir = Path(img_dir)
    frames = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(frames, key=natural_key)


def list_sequences(root: PathLike, only: Optional[Iterable[str]] = None) -> List[Path]:
    """Sequence folders under ``root``: every sub-directory containing an ``img1`` folder.

    ``only`` restricts the result to the given sequence names (an unknown name is an error).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Sequence root not found: {root}")
    seqs = [d for d in root.iterdir() if d.is_dir() and (d / "img1").is_dir()]
    if only is not None:
        wanted = set(only)
        missing = sorted(wanted - {d.name for d in seqs})
        if missing:
            raise FileNotFoundError(f"Sequences not found in {root}: {missing}")
        seqs = [d for d in seqs if d.name in wanted]
    return sorted(seqs, key=natural_key)


# --------------------------------------------------------------------------- tracking files
def read_tracking_file(path: PathLike) -> pd.DataFrame:
    """Read a ``frame,id,x,y,w,h[,...]`` file into a DataFrame with :data:`TRACK_COLUMNS`."""
    path = Path(path)
    if path.stat().st_size == 0:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in TRACK_COLUMNS}).astype(
            {"frame_id": int, "track_id": int}
        )
    df = pd.read_csv(path, header=None, usecols=range(6), names=TRACK_COLUMNS)
    df["frame_id"] = df["frame_id"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df


def write_tracking_file(df: pd.DataFrame, path: PathLike) -> None:
    """Write tracking results in the contest format: ``frame_id,object_id,x,y,w,h``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[TRACK_COLUMNS].sort_values(["frame_id", "track_id"])
    out.to_csv(path, index=False, header=False, float_format="%.2f")


# --------------------------------------------------------------------------- SoccerNet metadata
_TRACKLET_RE = re.compile(r"^\s*trackletID_(\d+)\s*=\s*(.*)$")


def read_gameinfo_labels(ini_path: PathLike) -> Dict[int, str]:
    """Parse a SoccerNet ``gameinfo.ini`` into ``{tracklet_id: label}``.

    Labels look like ``"player team left"``, ``"referee"``, ``"ball"``. Anything after the first
    ``;`` (e.g. jersey number) is dropped. Returns an empty dict when the file does not exist.
    """
    ini_path = Path(ini_path)
    labels: Dict[int, str] = {}
    if not ini_path.is_file():
        return labels
    with ini_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = _TRACKLET_RE.match(line)
            if m:
                labels[int(m.group(1))] = m.group(2).split(";")[0].strip().lower()
    return labels
