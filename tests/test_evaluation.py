import pytest

from helpers import make_sequence
from soccer_tracker.evaluation import compute_nmae, format_report, read_behavior_file


def _gt_rows(n_frames=20, ids=(1, 2, 3)):
    rows = []
    for f in range(1, n_frames + 1):
        for k in ids:
            rows.append([f, k, 20 * k + f, 30, 40, 100, 1, -1, -1, -1])
    return rows


def _dataset(tmp_path, n_frames=20, behavior=True):
    behavior_rows = [[f, r, 2] for f in range(1, n_frames + 1) for r in (1, 2)] if behavior else None
    make_sequence(tmp_path / "gt", "1", n_frames, size=(320, 180), gt_rows=_gt_rows(n_frames), behavior_rows=behavior_rows)
    make_sequence(tmp_path / "gt", "2", n_frames, size=(320, 180), gt_rows=_gt_rows(n_frames), behavior_rows=behavior_rows)
    return tmp_path / "gt"


def _write_predictions(root, n_frames=20, dropped_id=None, beh_value=2):
    root.mkdir(parents=True, exist_ok=True)
    for seq in ("1", "2"):
        lines = [
            f"{f},{k},{20 * k + f},30,40,100"
            for f in range(1, n_frames + 1) for k in (1, 2, 3) if k != dropped_id
        ]
        (root / f"tracking_{seq}_27.txt").write_text("\n".join(lines) + "\n")
        (root / f"behavior_{seq}_27.txt").write_text(
            "\n".join(f"{f},{r},{beh_value}" for f in range(1, n_frames + 1) for r in (1, 2)) + "\n"
        )


def test_nmae_perfect_and_degraded(tmp_path):
    gt = _dataset(tmp_path)
    _write_predictions(tmp_path / "pred", beh_value=2)
    assert compute_nmae(gt, tmp_path / "pred", 27) == {"MAE": 0.0, "nMAE": 1.0}
    _write_predictions(tmp_path / "pred", beh_value=5)           # every count off by 3
    res = compute_nmae(gt, tmp_path / "pred", 27)
    assert res["MAE"] == pytest.approx(3.0) and res["nMAE"] == pytest.approx(0.7)


def test_nmae_saturates_at_zero_and_missing_entries_count_as_zero(tmp_path):
    gt = _dataset(tmp_path)
    _write_predictions(tmp_path / "pred", beh_value=40)
    assert compute_nmae(gt, tmp_path / "pred", 27)["nMAE"] == 0.0
    for seq in ("1", "2"):
        (tmp_path / "pred" / f"behavior_{seq}_27.txt").write_text("")   # nothing predicted -> 0 players everywhere
    assert compute_nmae(gt, tmp_path / "pred", 27)["MAE"] == pytest.approx(2.0)


def test_nmae_is_none_when_behaviour_files_are_missing(tmp_path):
    gt = _dataset(tmp_path, behavior=False)
    _write_predictions(tmp_path / "pred")
    assert compute_nmae(gt, tmp_path / "pred", 27) == {"MAE": None, "nMAE": None}


def test_read_behavior_file(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("1,1,3\n1,2,4.0\n\n")
    assert read_behavior_file(p) == {(1, 1): 3, (1, 2): 4}


def test_format_report_handles_missing_values():
    text = format_report({"HOTA": 0.675, "DetA": 0.8, "AssA": 0.57, "MAE": None, "nMAE": None, "PTBS": 0.675})
    assert "0.6750" in text and "n/a" in text


# ---------------------------------------------------------------- TrackEval (optional dependency)
def test_hota_of_perfect_predictions_is_one(tmp_path):
    pytest.importorskip("trackeval")
    from soccer_tracker.evaluation import evaluate

    gt = _dataset(tmp_path)
    _write_predictions(tmp_path / "pred")
    res = evaluate(gt, tmp_path / "pred", team_id=27)
    assert res["HOTA"] == pytest.approx(1.0) and res["DetA"] == pytest.approx(1.0) and res["AssA"] == pytest.approx(1.0)
    assert res["nMAE"] == 1.0 and res["PTBS"] == pytest.approx(2.0)


def test_hota_drops_when_a_player_is_missed(tmp_path):
    pytest.importorskip("trackeval")
    from soccer_tracker.evaluation import compute_hota05

    gt = _dataset(tmp_path)
    _write_predictions(tmp_path / "pred", dropped_id=3)
    res = compute_hota05(gt, tmp_path / "pred", 27)
    assert 0.4 < res["HOTA"] < 0.9 and res["DetA"] < 0.8


def test_only_restricts_evaluation_to_a_subset_of_sequences(tmp_path):
    pytest.importorskip("trackeval")
    from soccer_tracker.evaluation import compute_hota05
    from soccer_tracker.io_utils import list_sequences

    gt = _dataset(tmp_path)
    _write_predictions(tmp_path / "pred")
    (tmp_path / "pred" / "tracking_2_27.txt").unlink()          # no predictions for sequence 2
    with pytest.raises(FileNotFoundError):
        compute_hota05(gt, tmp_path / "pred", 27)
    assert compute_hota05(gt, tmp_path / "pred", 27, only=["1"])["HOTA"] == pytest.approx(1.0)
    assert [s.name for s in list_sequences(gt, only=["2"])] == ["2"]
    with pytest.raises(FileNotFoundError):
        list_sequences(gt, only=["does-not-exist"])


def test_ball_is_removed_from_the_ground_truth(tmp_path):
    """SoccerNet GT includes the ball as a tracklet; the detector never predicts it, so it must not count."""
    pytest.importorskip("trackeval")
    from soccer_tracker.evaluation import compute_hota05

    gt_rows = _gt_rows(20, ids=(1, 2)) + [[f, 9, 150, 150, 6, 6, 1, -1, -1, -1] for f in range(1, 21)]
    ini = "[Sequence]\ntrackletID_1= player team left;1\ntrackletID_2= player team right;2\ntrackletID_9= ball;1\n"
    for seq in ("1", "2"):
        make_sequence(tmp_path / "gt", seq, 20, size=(320, 180), gt_rows=gt_rows, ini=ini)
    pred = tmp_path / "pred"
    pred.mkdir()
    for seq in ("1", "2"):
        (pred / f"tracking_{seq}_27.txt").write_text(
            "\n".join(f"{f},{k},{20 * k + f},30,40,100" for f in range(1, 21) for k in (1, 2)) + "\n"
        )
    assert compute_hota05(tmp_path / "gt", pred, 27, exclude_ball=True)["HOTA"] == pytest.approx(1.0)
    assert compute_hota05(tmp_path / "gt", pred, 27, exclude_ball=False)["HOTA"] < 0.95


def test_ablation_scores_every_variant(tmp_path):
    pytest.importorskip("trackeval")
    from soccer_tracker.ablation import default_variants, format_markdown, run_ablation

    gt = _dataset(tmp_path)
    _write_predictions(tmp_path / "pred")
    rows = run_ablation(gt, tmp_path / "pred", team_id=27)
    assert [r["variant"] for r in rows] == list(default_variants())
    assert all(0.0 <= r["HOTA"] <= 1.0 for r in rows)
    assert rows[0]["HOTA"] == pytest.approx(1.0)          # raw predictions are perfect
    table = format_markdown(rows, n_sequences=2)
    assert table.count("\n| ") == len(rows) and "_2 sequences._" in table
