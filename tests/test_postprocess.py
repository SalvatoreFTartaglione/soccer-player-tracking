import numpy as np
import pandas as pd
import pytest

from helpers import concat, make_track
from soccer_tracker.io_utils import read_tracking_file, write_tracking_file
from soccer_tracker.postprocess import (
    PostprocessConfig,
    interpolate_and_smooth,
    postprocess_dataframe,
    postprocess_folder,
    relabel_ids,
    remove_short_tracks,
    stitch_tracklets,
)


# ------------------------------------------------------------------ stitching
def test_stitch_merges_two_fragments_of_one_player():
    a = make_track(1, 1, 50, x0=100, vx=2)                    # ends at x = 198
    b = make_track(2, 60, 120, x0=100 + 2 * 59, vx=2)         # continues the same trajectory
    out = stitch_tracklets(concat(a, b))
    assert out["track_id"].nunique() == 1


def test_stitch_merges_chains_of_three_or_more_fragments():
    """Regression: the original implementation only ever merged pairs, never a -> b -> c."""
    a = make_track(1, 1, 30, x0=100, vx=2)
    b = make_track(2, 40, 70, x0=100 + 2 * 39, vx=2)
    c = make_track(3, 80, 110, x0=100 + 2 * 79, vx=2)
    d = make_track(4, 120, 150, x0=100 + 2 * 119, vx=2)
    out = stitch_tracklets(concat(a, b, c, d))
    assert out["track_id"].nunique() == 1
    assert set(out["track_id"]) == {1}  # the chain keeps the id of its first tracklet


def test_stitch_keeps_distinct_players_apart():
    a = make_track(1, 1, 50, x0=100, y0=300, vx=0)
    far = make_track(2, 60, 120, x0=1500, y0=300, vx=0)       # far away: not the same player
    out = stitch_tracklets(concat(a, far))
    assert out["track_id"].nunique() == 2


def test_stitch_respects_time_gap_limit():
    a = make_track(1, 1, 50, vx=0)
    late = make_track(2, 200, 260, vx=0)                        # same place but 150 frames later
    assert stitch_tracklets(concat(a, late), max_time_gap=50)["track_id"].nunique() == 2
    assert stitch_tracklets(concat(a, late), max_time_gap=200)["track_id"].nunique() == 1


def test_stitch_never_merges_tracks_that_coexist_in_time():
    a = make_track(1, 1, 100, x0=100, vx=0)
    b = make_track(2, 1, 100, x0=105, vx=0)                     # co-occurs with a: two different people
    out = stitch_tracklets(concat(a, b))
    assert out["track_id"].nunique() == 2


def test_stitch_assigns_each_successor_once():
    """Two ending tracks compete for one new track: only one of them may claim it."""
    a1 = make_track(1, 1, 50, x0=100, y0=300, vx=0)
    a2 = make_track(2, 1, 50, x0=130, y0=300, vx=0)
    nxt = make_track(3, 60, 120, x0=115, y0=300, vx=0)
    out = stitch_tracklets(concat(a1, a2, nxt))
    assert out["track_id"].nunique() == 2
    assert not out.duplicated(["frame_id", "track_id"]).any()


def test_stitch_does_not_modify_input_and_handles_empty():
    df = concat(make_track(1, 1, 20), make_track(2, 30, 60))
    snapshot = df.copy()
    stitch_tracklets(df)
    pd.testing.assert_frame_equal(df, snapshot)
    assert stitch_tracklets(df.iloc[0:0]).empty


# ------------------------------------------------------------------ interpolation / smoothing
def test_interpolation_fills_short_gap_linearly():
    df = make_track(1, 1, 30, x0=0, vx=10, skip=range(11, 16))   # frames 11..15 missing (gap of 5)
    out = interpolate_and_smooth(df, max_gap=10, sigma=0)
    assert sorted(out["frame_id"]) == list(range(1, 31))
    row = out.loc[out["frame_id"] == 13].iloc[0]
    assert row["x1"] == pytest.approx(120.0)


def test_interpolation_leaves_long_gaps_empty():
    """Regression: pandas' ``limit`` used to half-fill long gaps (hallucinated boxes)."""
    df = make_track(1, 1, 80, skip=range(21, 61))                # 40 missing frames
    out = interpolate_and_smooth(df, max_gap=10, sigma=0)
    frames = set(out["frame_id"])
    assert frames == set(range(1, 21)) | set(range(61, 81))
    assert not frames & set(range(21, 61))


def test_smoothing_does_not_bleed_across_long_gaps():
    left = make_track(1, 1, 20, x0=0, vx=0, vy=0)
    right = make_track(1, 61, 80, x0=1000, vx=0, vy=0)
    out = interpolate_and_smooth(concat(left, right), max_gap=10, sigma=3)
    # boxes right at the edges of the gap keep their own position (no averaging with the other side)
    assert out.loc[out["frame_id"] == 20, "x1"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert out.loc[out["frame_id"] == 61, "x1"].iloc[0] == pytest.approx(1000.0, abs=1e-6)


def test_smoothing_reduces_jitter_but_keeps_trend():
    rng = np.random.RandomState(0)
    df = make_track(1, 1, 100, x0=0, vx=5)
    noisy = df.copy()
    noisy["x1"] = noisy["x1"] + rng.normal(0, 4, len(noisy))
    out = interpolate_and_smooth(noisy, max_gap=10, sigma=2)
    err_before = np.abs(noisy["x1"] - df["x1"]).mean()
    err_after = np.abs(out["x1"].to_numpy() - df["x1"].to_numpy()).mean()
    assert err_after < err_before


def test_interpolation_keeps_integer_ids_and_handles_empty():
    out = interpolate_and_smooth(make_track(7, 1, 5))
    assert out["track_id"].dtype.kind == "i"
    assert interpolate_and_smooth(out.iloc[0:0]).empty


# ------------------------------------------------------------------ filtering / relabel
def test_remove_short_tracks():
    df = concat(make_track(1, 1, 60), make_track(2, 1, 10))
    out = remove_short_tracks(df, min_len=50)
    assert set(out["track_id"]) == {1}


def test_relabel_ids_starts_at_one_in_order_of_appearance():
    df = concat(make_track(42, 10, 30), make_track(7, 1, 30), make_track(99, 20, 40))
    out = relabel_ids(df)
    first_frame = out.groupby("track_id")["frame_id"].min()
    assert list(out["track_id"].unique().astype(int)) and sorted(first_frame.index) == [1, 2, 3]
    assert list(first_frame.values) == sorted(first_frame.values)


# ------------------------------------------------------------------ pipeline
def test_full_pipeline_output_is_clean():
    fragments = concat(
        make_track(5, 1, 60, x0=100, vx=2),
        make_track(9, 70, 140, x0=100 + 2 * 69, vx=2),   # same player after an occlusion
        make_track(11, 1, 8, x0=1500, y0=100, vx=0),      # ghost
    )
    out = postprocess_dataframe(fragments, PostprocessConfig())
    assert set(out["track_id"]) == {1}
    assert not out.duplicated(["frame_id", "track_id"]).any()
    assert out["frame_id"].is_monotonic_increasing
    # the 9-frame occlusion between the two fragments was bridged by interpolation
    assert set(range(1, 141)) <= set(out["frame_id"])


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        PostprocessConfig.from_dict({"stitch_max_gap": 3})


def test_postprocess_folder_ignores_behaviour_files(tmp_path):
    """Regression: ``*.txt`` used to match ``behavior_*.txt`` and corrupt them."""
    write_tracking_file(make_track(1, 1, 80), tmp_path / "tracking_1_27.txt")
    behavior = tmp_path / "behavior_1_27.txt"
    behavior.write_text("1,1,3\n1,2,0\n")
    postprocess_folder(tmp_path)
    assert behavior.read_text() == "1,1,3\n1,2,0\n"
    assert len(read_tracking_file(tmp_path / "tracking_1_27.txt")) == 80
