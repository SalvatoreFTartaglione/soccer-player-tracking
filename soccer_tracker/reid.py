"""Re-identification: ReID crop dataset from SoccerNet GT and OSNet-AIN fine-tuning with Torchreid.

Two stages, both usable from the CLI:

* :func:`build_reid_dataset` - crops every ground-truth box (every ``sample_every``-th frame) into
  ``<out>/train/<seq>_<id>/*.jpg`` and builds a query/gallery validation split from held-out sequences.
* :func:`fine_tune_reid` - fine-tunes ``osnet_ain_x1_0`` (ImageNet init) with triplet + label-smoothed
  cross-entropy loss, then :func:`export_reid_weights` extracts a clean ``state_dict`` that BoxMOT can load.

``torchreid`` is only needed for fine-tuning (not for inference) and is imported lazily.
"""

from __future__ import annotations

import csv
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from soccer_tracker.io_utils import read_gameinfo_labels

logger = logging.getLogger("soccer_tracker")

REID_HEIGHT = 256
REID_WIDTH = 128
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class ReIDDatasetConfig:
    sample_every: int = 10     # keep one frame out of N (consecutive frames are near-identical)
    min_bbox_area: int = 1000  # px^2: smaller boxes are unrecognisable blobs
    max_aspect: float = 1.25   # discard boxes wider than 1.25x their height (not upright persons)
    val_ratio: float = 0.10    # fraction of *sequences* held out for query/gallery validation
    seed: int = 42
    ignore_labels: Tuple[str, ...] = ("ball",)  # tracklet labels excluded from the identities


def _ignored_ids(seq_dir: Path, ignore_labels: Tuple[str, ...]) -> Set[int]:
    labels = read_gameinfo_labels(seq_dir / "gameinfo.ini")
    return {tid for tid, lbl in labels.items() if any(tok in lbl for tok in ignore_labels)}


def _read_gt(gt_path: Path, ignored: Set[int], sample_every: int) -> Dict[int, list]:
    """``{frame: [(track_id, x, y, w, h), ...]}`` for sampled frames, without ignored tracklets."""
    gt: Dict[int, list] = defaultdict(list)
    if not gt_path.is_file():
        return gt
    with gt_path.open(newline="") as fh:
        for row in csv.reader(fh):
            try:
                frame, tid = int(row[0]), int(row[1])
                x, y, w, h = map(float, row[2:6])
            except (ValueError, IndexError):
                logger.debug("Skipping malformed GT row: %s", row)
                continue
            if frame % sample_every != 0 or tid in ignored:
                continue
            gt[frame].append((tid, x, y, w, h))
    return gt


def extract_sequence_crops(seq_dir: Path, out_dir: Path, cfg: ReIDDatasetConfig) -> int:
    """Save the ReID crops of one sequence under ``out_dir/<seq>_<id>/``. Returns the number of crops."""
    import cv2

    gt = _read_gt(seq_dir / "gt" / "gt.txt", _ignored_ids(seq_dir, cfg.ignore_labels), cfg.sample_every)
    if not gt:
        logger.warning("No usable GT for %s - skipped", seq_dir.name)
        return 0

    saved = 0
    for frame_id in sorted(gt):
        frame_path = seq_dir / "img1" / f"{frame_id:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.debug("Cannot read %s", frame_path)
            continue
        height, width = frame.shape[:2]
        for tid, x, y, w, h in gt[frame_id]:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(width, int(x + w)), min(height, int(y + h))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0 or bw * bh < cfg.min_bbox_area or bw > bh * cfg.max_aspect:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], (REID_WIDTH, REID_HEIGHT))
            identity = f"{seq_dir.name}_{tid:04d}"
            (out_dir / identity).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / identity / f"{identity}_f{frame_id:06d}.jpg"), crop)
            saved += 1
    return saved


def build_reid_dataset(
    soccernet_tracking_root: Union[str, Path],
    output_root: Union[str, Path] = "reid_dataset",
    cfg: Optional[ReIDDatasetConfig] = None,
    overwrite: bool = False,
) -> Dict[str, int]:
    """Build ``train/``, ``query/`` and ``gallery/`` folders of identity crops from SoccerNet *train* sequences.

    A fraction ``cfg.val_ratio`` of the sequences is held out; inside them, one random crop per identity
    becomes the query and the remaining crops the gallery. Note that query and gallery of an identity come
    from the same tracklet (the SoccerNet GT has no cross-camera identities), so the resulting Rank-1/mAP
    are an optimistic proxy for real re-identification quality.
    """
    cfg = cfg or ReIDDatasetConfig()
    root = Path(soccernet_tracking_root) / "train"
    out = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"SoccerNet training data not found: {root}")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (pass overwrite=True to rebuild it)")
        shutil.rmtree(out)

    rng = random.Random(cfg.seed)
    sequences = sorted(p for p in root.iterdir() if p.is_dir())
    rng.shuffle(sequences)
    n_val = int(len(sequences) * cfg.val_ratio)
    val_seqs, train_seqs = sequences[:n_val], sequences[n_val:]
    logger.info("ReID dataset: %d train / %d validation sequences", len(train_seqs), len(val_seqs))

    stats = {"train": 0, "query": 0, "gallery": 0}
    for seq in train_seqs:
        stats["train"] += extract_sequence_crops(seq, out / "train", cfg)

    for seq in val_seqs:
        tmp = out / f"_tmp_{seq.name}"
        if extract_sequence_crops(seq, tmp, cfg) == 0:
            continue
        for ident_dir in sorted(tmp.iterdir()):
            imgs = sorted(f for f in ident_dir.iterdir() if f.suffix.lower() in _IMG_EXT)
            if len(imgs) < 2:  # need at least one query and one gallery image
                continue
            rng.shuffle(imgs)
            (out / "query" / ident_dir.name).mkdir(parents=True, exist_ok=True)
            (out / "gallery" / ident_dir.name).mkdir(parents=True, exist_ok=True)
            shutil.move(str(imgs[0]), str(out / "query" / ident_dir.name / imgs[0].name))
            stats["query"] += 1
            for img in imgs[1:]:
                shutil.move(str(img), str(out / "gallery" / ident_dir.name / img.name))
                stats["gallery"] += 1
        shutil.rmtree(tmp)

    logger.info("ReID dataset ready in %s: %s", out, stats)
    return stats


def check_alignment(dataset_root: Union[str, Path] = "reid_dataset") -> bool:
    """True if every query identity also has gallery images and no identity folder is empty."""
    root = Path(dataset_root)
    query = {p.name for p in (root / "query").iterdir() if p.is_dir()} if (root / "query").is_dir() else set()
    gallery = {p.name for p in (root / "gallery").iterdir() if p.is_dir()} if (root / "gallery").is_dir() else set()
    missing = query - gallery
    empty = [d for split in ("query", "gallery") for d in (root / split).glob("*") if d.is_dir() and not any(d.iterdir())]
    if missing:
        logger.error("%d query identities have no gallery images, e.g. %s", len(missing), sorted(missing)[:5])
    if empty:
        logger.error("%d empty identity folders, e.g. %s", len(empty), empty[:5])
    return not missing and not empty


# --------------------------------------------------------------------------- torchreid training
def _make_dataset_class():
    """Define the Torchreid dataset lazily so that importing this module does not need ``torchreid``."""
    import torchreid

    class SoccerNetReID(torchreid.data.ImageDataset):
        """Reads the ``train/query/gallery`` identity folders written by :func:`build_reid_dataset`."""

        def __init__(self, root: str = "", **kwargs):
            self.root = str(Path(root).expanduser().resolve())
            dirs = {name: Path(self.root) / name for name in ("train", "query", "gallery")}
            for name, d in dirs.items():
                if not d.is_dir():
                    raise RuntimeError(f"Missing folder: {d}")
            train = self._scan(dirs["train"], camid=0)
            # query and gallery must share one identity -> label mapping
            all_ids = sorted({p.name for d in (dirs["query"], dirs["gallery"]) for p in d.iterdir() if p.is_dir()})
            labels = {pid: i for i, pid in enumerate(all_ids)}
            query = self._scan(dirs["query"], camid=0, pid2label=labels)
            gallery = self._scan(dirs["gallery"], camid=1, pid2label=labels)  # different camid: same-camid matches are not counted
            super().__init__(train, query, gallery, **kwargs)

        @staticmethod
        def _scan(folder: Path, camid: int, pid2label: Optional[Dict[str, int]] = None) -> List[tuple]:
            files = sorted(f for f in folder.glob("*/*") if f.suffix.lower() in _IMG_EXT)
            if pid2label is None:
                pid2label = {pid: i for i, pid in enumerate(sorted({f.parent.name for f in files}))}
            return [(str(f), pid2label[f.parent.name], camid) for f in files]

    return SoccerNetReID


def fine_tune_reid(
    dataset_root: Union[str, Path] = "reid_dataset",
    pretrained_weights: Union[str, Path] = "osnet_ain_x1_0_imagenet.pth",
    save_dir: Union[str, Path] = "log/osnet_soccer",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    margin: float = 0.3,
    lr_step: int = 15,
    num_instances: int = 4,
    eval_freq: int = 5,
) -> None:
    """Fine-tune OSNet-AIN-x1.0 on the soccer ReID crops (triplet loss + label-smoothed cross-entropy)."""
    import torch
    import torchreid

    pretrained_weights = Path(pretrained_weights)
    if not pretrained_weights.is_file():
        raise FileNotFoundError(
            f"ImageNet weights not found: {pretrained_weights} (see the Torchreid model zoo for osnet_ain_x1_0)"
        )

    torchreid.data.register_image_dataset("soccer_custom", _make_dataset_class())
    datamanager = torchreid.data.ImageDataManager(
        root=str(dataset_root),
        sources="soccer_custom",
        targets="soccer_custom",
        height=REID_HEIGHT,
        width=REID_WIDTH,
        batch_size_train=batch_size,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
        num_instances=num_instances,
        train_sampler="RandomIdentitySampler",
    )
    logger.info("Training identities: %d (%d batches/epoch)", datamanager.num_train_pids, len(datamanager.train_loader))

    model = torchreid.models.build_model(
        name="osnet_ain_x1_0", num_classes=datamanager.num_train_pids, loss="triplet", pretrained=False
    )
    torchreid.utils.load_pretrained_weights(model, str(pretrained_weights))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=lr)
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler="single_step", stepsize=lr_step)
    engine = torchreid.engine.ImageTripletEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, margin=margin, weight_t=1.0, weight_x=1.0, label_smooth=True
    )
    engine.run(save_dir=str(save_dir), max_epoch=epochs, eval_freq=eval_freq, print_freq=10, test_only=False)


def export_reid_weights(checkpoint: Union[str, Path], output: Union[str, Path]) -> Path:
    """Extract the bare ``state_dict`` from a Torchreid checkpoint (``model.pth.tar-<epoch>``) for BoxMOT.

    Only load checkpoints you trained yourself: ``torch.load`` unpickles arbitrary objects.
    """
    import torch

    checkpoint, output = Path(checkpoint), Path(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output)
    logger.info("ReID weights exported: %s -> %s", checkpoint, output)
    return output
