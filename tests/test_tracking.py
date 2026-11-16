from pathlib import Path

import numpy as np
import pytest

from helpers import make_sequence
from soccer_tracker.io_utils import frame_number, list_frames, natural_key, read_tracking_file, write_tracking_file
from soccer_tracker.tracking import BOTSORT_KEYS, TrackingConfig, load_tracker_config, run_tracker

REPO = Path(__file__).resolve().parent.parent


class StubTracker:
    """Records every update call; echoes one track per detection (id = detection index + 1)."""

    def __init__(self):
        self.calls = []

    def update(self, dets, img):
        self.calls.append(len(dets))
        if len(dets) == 0:
            return np.empty((0,))  # BoxMOT returns an empty 1-D array when nothing is tracked
        return np.array([[*d[:4], i + 1, d[4], d[5], i] for i, d in enumerate(dets)])


def test_run_tracker_updates_on_every_frame_including_empty_ones():
    """Regression: frames without detections used to be skipped, so lost tracks never aged."""
    stub = StubTracker()
    box = np.array([[10, 20, 60, 220, 0.9, 0]], dtype=np.float32)
    detections = [(1, box, None), (2, np.empty((0, 6)), None), (3, box, None)]
    df = run_tracker(detections, stub)
    assert stub.calls == [1, 0, 1]
    assert list(df["frame_id"]) == [1, 3]
    row = df.iloc[0]
    assert (row.x1, row.y1, row.w, row.h) == (10, 20, 50, 200)   # xyxy -> xywh
    assert df["track_id"].dtype.kind == "i"


def test_run_tracker_with_no_tracks_returns_typed_empty_frame():
    df = run_tracker([(1, np.empty((0, 6)), None)], StubTracker())
    assert df.empty and list(df.columns) == ["frame_id", "track_id", "x1", "y1", "w", "h"]


def test_tracking_file_roundtrip_matches_contest_format(tmp_path):
    df = run_tracker([(1, np.array([[914.5, 855, 969.5, 1027, 0.8, 0]], dtype=np.float32), None)], StubTracker())
    path = tmp_path / "tracking_1_27.txt"
    write_tracking_file(df, path)
    assert path.read_text() == "1,1,914.50,855.00,55.00,172.00\n"
    pd_back = read_tracking_file(path)
    assert pd_back.iloc[0].w == 55.0


def test_empty_tracking_file_can_be_read(tmp_path):
    path = tmp_path / "tracking_1_27.txt"
    write_tracking_file(run_tracker([(1, np.empty((0, 6)), None)], StubTracker()), path)
    assert read_tracking_file(path).empty


def test_frames_are_sorted_naturally(tmp_path):
    seq = make_sequence(tmp_path, "1", n_frames=0)
    for n in (10, 2, 1):
        (seq / "img1" / f"{n}.jpg").write_bytes(b"x")
    assert [p.name for p in list_frames(seq / "img1")] == ["1.jpg", "2.jpg", "10.jpg"]
    assert frame_number("000123.jpg") == 123
    assert sorted(["b10", "b2"], key=natural_key) == ["b2", "b10"]


def test_repo_tracker_config_only_uses_known_keys():
    cfg = load_tracker_config(REPO / "configs" / "botsort_soccer.yaml")
    assert set(cfg) <= set(BOTSORT_KEYS)
    assert cfg["cmc_method"] == "sof" and cfg["with_reid"] is True


def test_pipeline_yaml_loads():
    cfg = TrackingConfig.from_pipeline_yaml(REPO / "configs" / "pipeline.yaml")
    assert cfg.imgsz == 1920 and cfg.team_id == 27 and cfg.postprocess.min_track_len == 50


# ---------------------------------------------------------------- integration with the real stack (optional)
@pytest.mark.skipif(
    not (REPO / "weights" / "osnet_ain_x1_0_soccer.pt").is_file(), reason="released ReID weights not available"
)
def test_botsort_with_released_reid_weights_on_fake_detections():
    pytest.importorskip("boxmot")
    from soccer_tracker.tracking import build_tracker

    tracker = build_tracker(
        load_tracker_config(REPO / "configs" / "botsort_soccer.yaml"),
        REPO / "weights" / "osnet_ain_x1_0_soccer.pt",
        device="cpu",
    )
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    frames = []
    for f in range(1, 8):
        dets = np.array(
            [[100 + 5 * f, 300, 160 + 5 * f, 480, 0.9, 0], [900 - 4 * f, 400, 960 - 4 * f, 590, 0.6, 0]], dtype=np.float32
        )
        if f == 4:
            dets = np.empty((0, 6), dtype=np.float32)   # a frame without detections must not crash the tracker
        frames.append((f, dets, img))
    df = run_tracker(frames, tracker)
    assert not df.empty
    assert df["track_id"].min() >= 1
    assert set(df["frame_id"]) <= set(range(1, 8)) and 4 not in set(df["frame_id"])
