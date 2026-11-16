import pandas as pd
import pytest

from helpers import make_sequence
from soccer_tracker.cli import build_parser, main
from soccer_tracker.io_utils import TRACK_COLUMNS, write_tracking_file


def test_every_subcommand_has_help():
    parser = build_parser()
    for cmd in ("prepare-data", "check-data", "train-detector", "val-detector", "plot-training", "build-reid-data",
                "train-reid", "export-reid", "run", "evaluate", "ablate", "render", "view"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([cmd, "--help"])
        assert exc.value.code == 0


def test_cli_reports_missing_inputs_without_traceback(tmp_path, capsys):
    assert main(["evaluate", "--sequences", str(tmp_path / "nope")]) == 1


def test_render_video_headless(tmp_path):
    cv2 = pytest.importorskip("cv2")
    roi = {"roi1": {"x": 0, "y": 0, "width": 0.5, "height": 1}, "roi2": {"x": 0.5, "y": 0, "width": 0.5, "height": 1}}
    seq = make_sequence(tmp_path, "1", n_frames=5, size=(320, 180), roi=roi)
    write_tracking_file(pd.DataFrame([(f, 1, 20 + f, 40, 30, 90) for f in range(1, 6)], columns=TRACK_COLUMNS), tmp_path / "t.txt")
    out = main(["render", "--frames", str(seq / "img1"), "--tracking", str(tmp_path / "t.txt"),
                "--roi", str(seq / "roi.json"), "--out", str(tmp_path / "out.mp4"), "--fps", "5"])
    assert out == 0
    cap = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()


def test_plot_training_curves(tmp_path):
    pytest.importorskip("matplotlib")
    from pathlib import Path

    from soccer_tracker.detection import best_epoch, load_results_csv, plot_training_metrics

    csv = Path(__file__).resolve().parent.parent / "results" / "detector_training" / "results.csv"
    if not csv.is_file():
        pytest.skip("results.csv not available")
    png = plot_training_metrics(csv, tmp_path / "curves.png")
    assert png.is_file() and png.stat().st_size > 10_000
    assert int(best_epoch(load_results_csv(csv))["epoch"]) == 7   # highest fitness = the epoch saved as best.pt
