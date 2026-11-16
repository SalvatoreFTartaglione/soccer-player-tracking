"""Post-processing ablation: re-evaluate the raw tracker output under different post-processing settings.

The raw output (``<out>/raw/``, written by ``run``) is re-processed in a temporary folder for every variant
and scored with TrackEval, so the contribution of each step (stitching, interpolation, smoothing, filtering)
can be read off without running the detector and tracker again.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from soccer_tracker.evaluation import compute_hota05
from soccer_tracker.io_utils import list_sequences, read_tracking_file, tracking_filename, write_tracking_file
from soccer_tracker.postprocess import PostprocessConfig, postprocess_dataframe

PathLike = Union[str, Path]


def default_variants(base: Optional[PostprocessConfig] = None) -> Dict[str, Optional[PostprocessConfig]]:
    """Name -> post-processing configuration. ``None`` means "no post-processing at all" (raw output)."""
    base = base or PostprocessConfig()
    off = dict(stitch=False, interpolate=False, min_track_len=0)
    return {
        "raw tracker output": None,
        "full post-processing": base,
        "stitching only": replace(base, **{**off, "stitch": True}),
        "interpolation + smoothing only": replace(base, **{**off, "interpolate": True}),
        "short-track filter only": replace(base, **{**off, "min_track_len": base.min_track_len}),
        "full, without smoothing": replace(base, smooth_sigma=0.0),
    }


def run_ablation(
    dataset_root: PathLike,
    raw_dir: PathLike,
    team_id: Union[int, str] = 27,
    only: Optional[Iterable[str]] = None,
    base: Optional[PostprocessConfig] = None,
    fps: float = 25.0,
    split: str = "test",
) -> List[Dict[str, float]]:
    """Evaluate every variant of :func:`default_variants`; returns one row ``{variant, HOTA, DetA, AssA}`` each."""
    seqs = [s.name for s in list_sequences(dataset_root, only)]
    rows = []
    with tempfile.TemporaryDirectory(prefix="soccer_ablation_") as tmp:
        for name, cfg in default_variants(base).items():
            if cfg is None:
                folder = Path(raw_dir)
            else:
                folder = Path(tmp) / name.replace(" ", "_").replace("+", "and").replace(",", "")
                for seq in seqs:
                    fname = tracking_filename(seq, team_id)
                    write_tracking_file(postprocess_dataframe(read_tracking_file(Path(raw_dir) / fname), cfg), folder / fname)
            metrics = compute_hota05(dataset_root, folder, team_id, fps, split, only=seqs)
            rows.append({"variant": name, **metrics})
    return rows


def format_markdown(rows: List[Dict[str, float]], n_sequences: Optional[int] = None) -> str:
    lines = ["| Variant | HOTA@0.5 | DetA | AssA |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['HOTA']:.4f} | {r['DetA']:.4f} | {r['AssA']:.4f} |")
    if n_sequences:
        lines.append("")
        lines.append(f"_{n_sequences} sequences._")
    return "\n".join(lines)
