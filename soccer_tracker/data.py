"""SoccerNet tracking data: download, conversion to YOLO format, sub-sampling and visual checks."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import yaml

from soccer_tracker.io_utils import read_gameinfo_labels

logger = logging.getLogger("soccer_tracker")

#: SoccerNet tracklet labels kept as ``Person`` (class 0). The ball is dropped.
PERSON_LABELS = frozenset(
    {
        "player team left",
        "player team right",
        "goalkeepers team left",
        "goalkeepers team right",
        "goalkeeper team left",
        "goalkeeper team right",
        "referee",
    }
)
CLASS_NAMES = ["Person"]


HF_TRACKING_REPO = "SoccerNet/SN-Tracking-2023"  # public mirror; the original KAUST share now answers 401
HF_TRACKING_FILES = {"train": "train.zip", "test": "test.zip", "challenge": "challenge2023.zip"}


def _flatten_sequences(split_dir: Path) -> int:
    """Make sure ``split_dir/SNMOT-*`` are direct children, whatever top-level folder the archive used."""
    for seq in sorted(split_dir.rglob("SNMOT*")):
        if seq.is_dir() and seq.parent != split_dir and (seq / "img1").is_dir():
            shutil.move(str(seq), str(split_dir / seq.name))
    return len(list(split_dir.glob("SNMOT*")))


def download_soccernet(local_dir: Union[str, Path] = "SoccerNetData", splits: Sequence[str] = ("train", "test")) -> Path:
    """Download and extract SoccerNet *tracking* splits into ``<local_dir>/tracking/<split>/SNMOT-*``.

    Uses the public Hugging Face mirror (train ~9.6 GB, test ~8.7 GB, challenge ~5.3 GB). The zip is
    deleted after extraction; already extracted splits are skipped.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    tracking_dir = Path(local_dir) / "tracking"
    for split in splits:
        if split not in HF_TRACKING_FILES:
            raise ValueError(f"Unknown split '{split}', expected one of {sorted(HF_TRACKING_FILES)}")
        split_dir = tracking_dir / split
        if split_dir.is_dir() and any(split_dir.glob("SNMOT*")):
            logger.info("Split '%s' already present in %s - skipped", split, split_dir)
            continue
        logger.info("Downloading %s from %s ...", HF_TRACKING_FILES[split], HF_TRACKING_REPO)
        zip_path = Path(
            hf_hub_download(
                repo_id=HF_TRACKING_REPO,
                repo_type="dataset",
                filename=HF_TRACKING_FILES[split],
                local_dir=str(tracking_dir / "_download"),
            )
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting %s ...", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(split_dir)
        n_seq = _flatten_sequences(split_dir)
        if n_seq == 0:
            raise RuntimeError(f"No SNMOT-* sequences found after extracting {zip_path.name} into {split_dir}")
        logger.info("Split '%s': %d sequences", split, n_seq)
        zip_path.unlink()
    shutil.rmtree(tracking_dir / "_download", ignore_errors=True)
    return tracking_dir


def to_yolo_bbox(box_xywh: Iterable[float], img_w: int, img_h: int) -> Optional[List[float]]:
    """Convert a top-left ``(x, y, w, h)`` pixel box to normalised YOLO ``(xc, yc, w, h)``.

    The box is clipped to the image first: SoccerNet annotates players that are partially out of
    frame, and Ultralytics discards whole images whose labels fall outside ``[0, 1]``. Returns
    ``None`` when nothing of the box remains inside the image.
    """
    x, y, w, h = (float(v) for v in box_xywh)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [
        (x1 + x2) / 2.0 / img_w,
        (y1 + y2) / 2.0 / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    ]


def _image_size(path: Path) -> Tuple[int, int]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"Cannot read image {path}")
    return img.shape[1], img.shape[0]


def prepare_yolo_dataset(
    train_dir: Union[str, Path],
    valid_dir: Union[str, Path],
    yolo_root: Union[str, Path],
    copy_images: bool = True,
) -> Path:
    """Convert SoccerNet tracking sequences (``SNMOT-*``) into a single-class YOLO dataset.

    Players, goalkeepers and referees all become class ``0`` (``Person``); the ball is ignored.
    Layout produced under ``yolo_root``::

        images/{train,valid}/<SEQ>_<frame>.jpg
        labels/{train,valid}/<SEQ>_<frame>.txt
        data.yaml

    Only frames with at least one person box are exported. Returns the path of ``data.yaml``.
    """
    yolo_root = Path(yolo_root)
    splits = {"train": Path(train_dir), "valid": Path(valid_dir)}
    for split, src_root in splits.items():
        if not any(src_root.glob("SNMOT*")):
            raise FileNotFoundError(f"No SNMOT-* sequences found for the '{split}' split in {src_root}")

    for split, src_root in splits.items():
        img_out = yolo_root / "images" / split
        lbl_out = yolo_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for seq_dir in sorted(src_root.glob("SNMOT*")):
            gt_file = seq_dir / "gt" / "gt.txt"
            if not gt_file.is_file():
                logger.warning("Skipping %s: no gt.txt", seq_dir.name)
                continue
            labels = read_gameinfo_labels(seq_dir / "gameinfo.ini")
            person_ids = {tid for tid, lbl in labels.items() if lbl in PERSON_LABELS}
            gt = pd.read_csv(
                gt_file, header=None, usecols=range(6), names=["frame", "id", "x", "y", "w", "h"]
            )
            gt = gt[gt["id"].isin(person_ids)]
            logger.info("%s/%s: %d person boxes", split, seq_dir.name, len(gt))

            size: Optional[Tuple[int, int]] = None  # all frames of a sequence share the same resolution
            for frame, frame_df in gt.groupby("frame"):
                src_img = seq_dir / "img1" / f"{int(frame):06d}.jpg"
                if not src_img.is_file():
                    continue
                if size is None:
                    size = _image_size(src_img)
                lines = []
                for x, y, w, h in frame_df[["x", "y", "w", "h"]].to_numpy():
                    box = to_yolo_bbox((x, y, w, h), *size)
                    if box is not None:
                        lines.append("0 " + " ".join(f"{v:.6f}" for v in box))
                if not lines:
                    continue
                stem = f"{seq_dir.name}_{int(frame):06d}"
                dst_img = img_out / f"{stem}.jpg"
                if not dst_img.exists():
                    if copy_images:
                        shutil.copy2(src_img, dst_img)
                    else:
                        dst_img.symlink_to(src_img.resolve())
                (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    data_yaml = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = yolo_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return yaml_path


def create_subsampled_list(yolo_root: Union[str, Path], split: str = "train", step: int = 4) -> Optional[Path]:
    """Write ``<split>_subset.txt`` listing every ``step``-th image (consecutive frames are near-duplicates)."""
    yolo_root = Path(yolo_root)
    images = sorted((yolo_root / "images" / split).glob("*.jpg"))
    if not images:
        logger.warning("No images found for split '%s' in %s", split, yolo_root)
        return None
    subset = images[::step]
    out = yolo_root / f"{split}_subset.txt"
    out.write_text("\n".join(str(p.resolve()) for p in subset) + "\n")
    logger.info("[%s] %d images -> %d (step %d)", split, len(images), len(subset), step)
    return out


def read_yolo_labels(label_file: Union[str, Path]) -> List[Tuple[int, float, float, float, float]]:
    """Read ``class xc yc w h`` rows from a YOLO label file."""
    rows = []
    for line in Path(label_file).read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def check_conversion(
    yolo_root: Union[str, Path],
    split: str = "train",
    num_samples: int = 20,
    shuffle: bool = True,
    seed: int = 0,
    show: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
) -> int:
    """Visual sanity check of the YOLO conversion: draw the converted boxes on sample images.

    Annotated images are saved to ``save_dir`` (if given) and/or shown in an OpenCV window
    (``show=True``; any key = next, ``q`` = quit). Returns the number of images processed.
    """
    import cv2

    yolo_root = Path(yolo_root)
    images = sorted((yolo_root / "images" / split).glob("*.jpg"))
    if shuffle:
        random.Random(seed).shuffle(images)
    images = images[:num_samples]
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    done = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read %s", img_path)
            continue
        h, w = img.shape[:2]
        label_file = yolo_root / "labels" / split / f"{img_path.stem}.txt"
        n_boxes = 0
        if label_file.is_file():
            for _, xc, yc, bw, bh in read_yolo_labels(label_file):
                p1 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
                p2 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
                cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
                n_boxes += 1
        cv2.putText(img, f"{img_path.name} | {n_boxes} boxes", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if save_dir:
            cv2.imwrite(str(Path(save_dir) / img_path.name), img)
        done += 1
        if show:
            cv2.imshow("YOLO conversion check (any key: next, q: quit)", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    if show:
        cv2.destroyAllWindows()
    return done
