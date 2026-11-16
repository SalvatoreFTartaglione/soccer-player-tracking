from pathlib import Path

import pytest
import yaml

from helpers import make_sequence
from soccer_tracker.data import check_conversion, create_subsampled_list, prepare_yolo_dataset, to_yolo_bbox
from soccer_tracker.io_utils import read_gameinfo_labels

INI = """[Sequence]
name=SNMOT-001
trackletID_1= player team left;10
trackletID_2= player team right;7
trackletID_3= ball;1
trackletID_4= referee;main
trackletID_5= goalkeepers team left;1
"""


def test_to_yolo_bbox_inside_image():
    xc, yc, w, h = to_yolo_bbox((100, 200, 50, 100), 1000, 1000)
    assert (xc, yc, w, h) == pytest.approx((0.125, 0.25, 0.05, 0.1))


def test_to_yolo_bbox_is_clipped_to_the_image():
    """Regression: boxes crossing the border produced labels > 1, which make Ultralytics drop the whole image."""
    xc, yc, w, h = to_yolo_bbox((-20, 900, 100, 200), 1000, 1000)   # 20 px left of, 100 px below the border
    assert 0 <= xc - w / 2 and xc + w / 2 <= 1 and 0 <= yc - h / 2 and yc + h / 2 <= 1
    assert w == pytest.approx(0.08) and h == pytest.approx(0.1)


def test_to_yolo_bbox_fully_outside_or_degenerate():
    assert to_yolo_bbox((-200, 10, 100, 50), 1000, 1000) is None
    assert to_yolo_bbox((10, 10, 0, 50), 1000, 1000) is None


def test_read_gameinfo_labels(tmp_path):
    p = tmp_path / "gameinfo.ini"
    p.write_text(INI)
    assert read_gameinfo_labels(p) == {
        1: "player team left", 2: "player team right", 3: "ball", 4: "referee", 5: "goalkeepers team left",
    }
    assert read_gameinfo_labels(tmp_path / "missing.ini") == {}


def _soccernet_like(root):
    gt = [
        [1, 1, 10, 20, 30, 80, 1, -1, -1, -1],
        [1, 3, 100, 100, 8, 8, 1, -1, -1, -1],      # ball -> must be dropped
        [1, 4, -15, 50, 40, 90, 1, -1, -1, -1],     # referee partially out of the frame
        [2, 3, 100, 100, 8, 8, 1, -1, -1, -1],      # frame 2 only has the ball -> no label file
    ]
    make_sequence(root, "SNMOT-001", n_frames=2, size=(320, 180), gt_rows=gt, ini=INI)


def test_prepare_yolo_dataset_end_to_end(tmp_path):
    _soccernet_like(tmp_path / "train")
    _soccernet_like(tmp_path / "test")
    yaml_path = prepare_yolo_dataset(tmp_path / "train", tmp_path / "test", tmp_path / "yolo")

    cfg = yaml.safe_load(yaml_path.read_text())
    assert cfg["nc"] == 1 and cfg["names"] == ["Person"] and cfg["val"] == "images/valid"

    for split in ("train", "valid"):
        labels = sorted((tmp_path / "yolo" / "labels" / split).glob("*.txt"))
        assert [p.name for p in labels] == ["SNMOT-001_000001.txt"]       # frame 2 (ball only) is not exported
        rows = [list(map(float, line.split())) for line in labels[0].read_text().splitlines()]
        assert len(rows) == 2                                            # player + referee, no ball
        assert all(r[0] == 0 and all(0 <= v <= 1 for v in r[1:]) for r in rows)
        assert (tmp_path / "yolo" / "images" / split / "SNMOT-001_000001.jpg").is_file()


def test_subsampled_list_and_visual_check(tmp_path):
    _soccernet_like(tmp_path / "train")
    _soccernet_like(tmp_path / "test")
    prepare_yolo_dataset(tmp_path / "train", tmp_path / "test", tmp_path / "yolo")
    lst = create_subsampled_list(tmp_path / "yolo", "train", step=2)
    assert lst is not None and len(lst.read_text().split()) == 1
    assert check_conversion(tmp_path / "yolo", "train", num_samples=5, save_dir=tmp_path / "viz") == 1
    assert (tmp_path / "viz" / "SNMOT-001_000001.jpg").is_file()


def test_prepare_yolo_dataset_fails_loudly_without_sources(tmp_path):
    """Regression: a failed download used to end with an empty dataset and a 'ready' message."""
    with pytest.raises(FileNotFoundError):
        prepare_yolo_dataset(tmp_path / "nope" / "train", tmp_path / "nope" / "test", tmp_path / "yolo")
    assert not (tmp_path / "yolo").exists()


@pytest.mark.parametrize("top_level", ["", "test/"])
def test_download_soccernet_extracts_and_flattens(tmp_path, monkeypatch, top_level):
    import zipfile

    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    from soccer_tracker.data import download_soccernet

    def fake_hf_hub_download(repo_id, filename, repo_type, local_dir):
        archive = Path(local_dir) / filename
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(f"{top_level}SNMOT-001/img1/000001.jpg", b"x")
            zf.writestr(f"{top_level}SNMOT-001/gt/gt.txt", "1,1,0,0,5,5,1,-1,-1,-1\n")
        return str(archive)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    out = download_soccernet(tmp_path / "SoccerNetData", splits=("test",))
    assert (out / "test" / "SNMOT-001" / "img1" / "000001.jpg").is_file()
    assert not (out / "_download").exists()                       # temporary zip folder removed
    download_soccernet(tmp_path / "SoccerNetData", splits=("test",))   # second call: already present, no download
    with pytest.raises(ValueError):
        download_soccernet(tmp_path / "SoccerNetData", splits=("valid",))
