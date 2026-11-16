import pandas as pd

from helpers import make_sequence
from soccer_tracker.behavior import (
    count_players_in_rois,
    generate_behavior_analysis,
    load_rois,
    roi_pixel_rect,
)
from soccer_tracker.io_utils import TRACK_COLUMNS, write_tracking_file

ROIS = {1: {"x": 0.0, "y": 0.5, "width": 0.5, "height": 0.5}, 2: {"x": 0.5, "y": 0.5, "width": 0.5, "height": 0.5}}


def boxes(rows):
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def test_roi_pixel_rect():
    assert roi_pixel_rect({"x": 0.25, "y": 0.5, "width": 0.5, "height": 0.25}, 1920, 1080) == (480.0, 540.0, 1440.0, 810.0)


def test_player_counted_by_base_centre_not_by_box_overlap():
    # box straddles the ROI but its feet (x + w/2, y + h) are outside (y + h = 400 < 540)
    tracks = boxes([(1, 1, 100, 100, 100, 300)])
    out = count_players_in_rois(tracks, ROIS, 1920, 1080, n_frames=1)
    assert out["n_players"].sum() == 0
    # same box moved down: feet at y = 600 inside the ROI
    tracks = boxes([(1, 1, 100, 300, 100, 300)])
    out = count_players_in_rois(tracks, ROIS, 1920, 1080, n_frames=1)
    assert out.query("region_id == 1")["n_players"].iloc[0] == 1
    assert out.query("region_id == 2")["n_players"].iloc[0] == 0


def test_roi_border_is_inclusive():
    # feet exactly on the ROI-1 left/bottom border: x + w/2 = 0, y + h = 540
    tracks = boxes([(1, 1, -50, 440, 100, 100)])
    out = count_players_in_rois(tracks, ROIS, 1920, 1080, n_frames=1)
    assert out.query("region_id == 1")["n_players"].iloc[0] == 1


def test_every_frame_and_region_is_reported_including_empty_ones():
    """Regression: frames without tracks used to be omitted instead of reported as 0."""
    tracks = boxes([(2, 1, 100, 700, 50, 100), (2, 2, 300, 700, 50, 100), (4, 3, 1200, 700, 50, 100)])
    out = count_players_in_rois(tracks, ROIS, 1920, 1080, n_frames=6)
    assert len(out) == 6 * 2
    assert out.query("frame_id == 2 and region_id == 1")["n_players"].iloc[0] == 2
    assert out.query("frame_id == 4 and region_id == 2")["n_players"].iloc[0] == 1
    assert out.query("frame_id == 5")["n_players"].sum() == 0
    assert list(out["frame_id"].unique()) == [1, 2, 3, 4, 5, 6]


def test_no_tracks_gives_zero_counts():
    out = count_players_in_rois(boxes([]), ROIS, 1920, 1080, n_frames=3)
    assert len(out) == 6 and out["n_players"].sum() == 0


def test_load_rois_maps_keys_to_region_ids(tmp_path):
    p = tmp_path / "roi.json"
    p.write_text('{"roi2": {"x": 0, "y": 0, "width": 1, "height": 1}, "roi1": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}}')
    rois = load_rois(p)
    assert sorted(rois) == [1, 2] and rois[1]["x"] == 0.1


def test_generate_behavior_analysis_end_to_end(tmp_path):
    roi = {"roi1": {"x": 0, "y": 0, "width": 0.5, "height": 1}, "roi2": {"x": 0.5, "y": 0, "width": 0.5, "height": 1}}
    make_sequence(tmp_path / "videos", "1", n_frames=4, size=(200, 100), roi=roi)
    pred = tmp_path / "pred"
    # frame 1: one player at the left half, frame 3: one in each half
    write_tracking_file(boxes([(1, 1, 10, 10, 20, 50), (3, 1, 10, 10, 20, 50), (3, 2, 150, 10, 20, 50)]), pred / "tracking_1_27.txt")
    written = generate_behavior_analysis(pred, tmp_path / "videos", team_id=27)
    assert [w.name for w in written] == ["behavior_1_27.txt"]
    lines = (pred / "behavior_1_27.txt").read_text().split()
    assert len(lines) == 4 * 2
    assert "1,1,1" in lines and "3,1,1" in lines and "3,2,1" in lines and "2,1,0" in lines


def test_generate_behavior_skips_sequences_without_roi(tmp_path):
    make_sequence(tmp_path / "videos", "1", n_frames=2)
    pred = tmp_path / "pred"
    write_tracking_file(boxes([(1, 1, 10, 10, 20, 50)]), pred / "tracking_1_27.txt")
    assert generate_behavior_analysis(pred, tmp_path / "videos", team_id=27) == []
