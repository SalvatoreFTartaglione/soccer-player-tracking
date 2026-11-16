"""Regenerate the figures used in the README from real results (no dataset frames are used).

    python scripts/make_readme_figures.py --sequences SoccerNetData/tracking/test \
        --predictions Predictions_test --sequence SNMOT-118

Outputs (in results/figures/): ablation.png, tracking_example.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from soccer_tracker.io_utils import read_gameinfo_labels, read_tracking_file, tracking_filename  # noqa: E402


def parse_ablation(md_path: Path):
    rows = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and re.match(r"^\d\.\d+$", cells[1]):
            rows.append((cells[0], *map(float, cells[1:])))
    return rows


def plot_ablation(md_path: Path, out: Path) -> None:
    rows = parse_ablation(md_path)
    base = rows[0]
    labels = [r[0] for r in rows[1:]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (name, color) in enumerate((("HOTA@0.5", "#1f77b4"), ("DetA", "#2ca02c"), ("AssA", "#ff7f0e"))):
        delta = [100 * (r[1 + k] - base[1 + k]) for r in rows[1:]]
        ax.bar(x + (k - 1) * 0.26, delta, 0.26, label=name, color=color)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [lab.replace(" only", "\nonly").replace(", without", ",\nwithout") for lab in labels], fontsize=9)
    ax.set_ylabel("change vs. raw tracker output (points)")
    ax.set_title("Post-processing ablation (14 SoccerNet test sequences)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_tracking_example(seq_dir: Path, pred_file: Path, out: Path) -> None:
    pred = read_tracking_file(pred_file)
    gt = read_tracking_file(seq_dir / "gt" / "gt.txt")
    ball = {i for i, lbl in read_gameinfo_labels(seq_dir / "gameinfo.ini").items() if "ball" in lbl}
    gt = gt[~gt["track_id"].isin(ball)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    cmap = plt.get_cmap("tab20")
    for i, (tid, g) in enumerate(pred.groupby("track_id")):
        g = g.sort_values("frame_id")
        ax1.plot(g["x1"] + g["w"] / 2, g["y1"] + g["h"], color=cmap(i % 20), lw=1.4)
        ax1.plot(g["x1"].iloc[-1] + g["w"].iloc[-1] / 2, g["y1"].iloc[-1] + g["h"].iloc[-1], "o", color=cmap(i % 20), ms=3)
    ax1.set_xlim(0, 1920)
    ax1.set_ylim(1080, 0)
    ax1.set_aspect("equal")
    ax1.set_title(f"{seq_dir.name}: foot-point trajectory of each predicted ID ({pred['track_id'].nunique()} IDs)")
    ax1.set_xlabel("x (px, camera moves)")
    ax1.set_ylabel("y (px)")

    frames = np.arange(1, int(max(pred["frame_id"].max(), gt["frame_id"].max())) + 1)
    ax2.plot(frames, gt.groupby("frame_id").size().reindex(frames, fill_value=0), label="ground truth", color="black", lw=1.5)
    ax2.plot(frames, pred.groupby("frame_id").size().reindex(frames, fill_value=0), label="prediction", color="#d62728", lw=1.2, ls="--")
    ax2.set_title("Players on screen per frame")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("count")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="SoccerNetData/tracking/test")
    ap.add_argument("--predictions", default="Predictions_test")
    ap.add_argument("--sequence", default="SNMOT-118")
    ap.add_argument("--team-id", type=int, default=27)
    ap.add_argument("--ablation", default="results/ablation.md")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_ablation(Path(a.ablation), out / "ablation.png")
    plot_tracking_example(
        Path(a.sequences) / a.sequence, Path(a.predictions) / tracking_filename(a.sequence, a.team_id), out / "tracking_example.png"
    )
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()
