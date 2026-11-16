import pytest

from helpers import make_sequence
from soccer_tracker.reid import ReIDDatasetConfig, build_reid_dataset, check_alignment

INI = "[Sequence]\ntrackletID_1= player team left;1\ntrackletID_2= player team right;2\ntrackletID_3= ball;1\n"


def _make_train_set(root, n_seq=4):
    # frames 10, 20, 30 are sampled (every 10th); each player appears in all of them
    gt = []
    for f in (10, 20, 30, 15):
        gt += [
            [f, 1, 20, 10, 40, 120, 1, -1, -1, -1],      # large player
            [f, 2, 100, 10, 40, 120, 1, -1, -1, -1],
            [f, 3, 150, 150, 6, 6, 1, -1, -1, -1],       # ball
        ]
    gt.append([10, 1, 200, 10, 5, 8, 1, -1, -1, -1])      # tiny box on frame 10 (area < 1000) for another id -> id 1 anyway
    for i in range(n_seq):
        make_sequence(root / "train", f"SNMOT-{i:03d}", n_frames=30, size=(320, 180), gt_rows=gt, ini=INI)


def test_build_reid_dataset_structure_and_filters(tmp_path):
    _make_train_set(tmp_path / "tracking", n_seq=5)
    cfg = ReIDDatasetConfig(sample_every=10, val_ratio=0.2, seed=1)
    stats = build_reid_dataset(tmp_path / "tracking", tmp_path / "reid", cfg)

    # 5 sequences, 1 held out: 4 train sequences * 2 identities * 3 sampled frames = 24 crops
    assert stats["train"] == 24
    identities = sorted(p.name for p in (tmp_path / "reid" / "train").iterdir())
    assert len(identities) == 8 and all(not name.endswith("0003") for name in identities)  # ball (id 3) excluded
    # validation: 2 identities -> 2 queries, the other 2 crops of each go to the gallery
    assert stats["query"] == 2 and stats["gallery"] == 4
    assert check_alignment(tmp_path / "reid")
    assert not list((tmp_path / "reid").glob("_tmp_*"))   # temp folders cleaned up


def test_build_reid_dataset_is_deterministic(tmp_path):
    _make_train_set(tmp_path / "tracking", n_seq=5)
    cfg = ReIDDatasetConfig(val_ratio=0.2, seed=7)
    build_reid_dataset(tmp_path / "tracking", tmp_path / "a", cfg)
    build_reid_dataset(tmp_path / "tracking", tmp_path / "b", cfg)
    def names(root):
        return sorted(str(p.relative_to(root)) for p in root.rglob("*.jpg"))

    assert names(tmp_path / "a") == names(tmp_path / "b")


def test_build_reid_dataset_refuses_to_overwrite(tmp_path):
    _make_train_set(tmp_path / "tracking", n_seq=2)
    build_reid_dataset(tmp_path / "tracking", tmp_path / "reid", ReIDDatasetConfig(val_ratio=0.0))
    with pytest.raises(FileExistsError):
        build_reid_dataset(tmp_path / "tracking", tmp_path / "reid")
    build_reid_dataset(tmp_path / "tracking", tmp_path / "reid", ReIDDatasetConfig(val_ratio=0.0), overwrite=True)
