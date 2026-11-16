"""Command line interface: ``python -m soccer_tracker <command> --help``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Anaconda environments often ship two OpenMP runtimes (torch + MKL); this avoids the hard abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger("soccer_tracker")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_prepare_data(a: argparse.Namespace) -> None:
    from soccer_tracker import data

    root = Path(a.soccernet_root)
    if a.download:
        data.download_soccernet(root, splits=tuple(a.splits))
    tracking = root / "tracking"
    if not ((tracking / "train").is_dir() and (tracking / "test").is_dir()):
        logger.info("Both 'train' and 'test' splits are needed for the YOLO conversion: nothing converted.")
        return
    yaml_path = data.prepare_yolo_dataset(tracking / "train", tracking / "test", a.out, copy_images=not a.symlink)
    logger.info("YOLO dataset ready: %s", yaml_path)
    if a.subsample_step:
        for split in ("train", "valid"):
            data.create_subsampled_list(a.out, split, a.subsample_step)


def _cmd_check_data(a: argparse.Namespace) -> None:
    from soccer_tracker import data

    n = data.check_conversion(a.yolo_root, a.split, a.num_samples, show=a.show, save_dir=a.save_dir)
    logger.info("Checked %d images", n)


def _cmd_train_detector(a: argparse.Namespace) -> None:
    from soccer_tracker import detection

    detection.train_detector(a.data, a.config, a.project, a.name, resume=a.resume, device=a.device)


def _cmd_val_detector(a: argparse.Namespace) -> None:
    from soccer_tracker import detection

    for k, v in detection.validate_detector(a.weights, a.data, a.imgsz, a.conf, a.iou, a.split, a.device).items():
        print(f"{k:>10}: {v:.4f}")


def _cmd_plot_training(a: argparse.Namespace) -> None:
    from soccer_tracker import detection

    print(detection.plot_training_metrics(a.csv, a.out))


def _cmd_build_reid(a: argparse.Namespace) -> None:
    from soccer_tracker import reid

    cfg = reid.ReIDDatasetConfig(sample_every=a.sample_every)
    print(reid.build_reid_dataset(a.soccernet_tracking_root, a.out, cfg, overwrite=a.overwrite))


def _cmd_train_reid(a: argparse.Namespace) -> None:
    from soccer_tracker import reid

    reid.fine_tune_reid(a.dataset, a.pretrained, a.save_dir, a.epochs, a.batch_size, a.lr)


def _cmd_export_reid(a: argparse.Namespace) -> None:
    from soccer_tracker import reid

    reid.export_reid_weights(a.checkpoint, a.out)


def _cmd_run(a: argparse.Namespace) -> None:
    from soccer_tracker.behavior import generate_behavior_analysis
    from soccer_tracker.tracking import TrackingConfig, track_sequences

    cfg = TrackingConfig.from_pipeline_yaml(a.config)
    if a.team_id is not None:
        cfg.team_id = a.team_id
    if a.device:
        cfg.device = a.device
    track_sequences(a.sequences, a.out, cfg, postprocess=not a.no_postprocess, only=a.only)
    generate_behavior_analysis(a.out, a.sequences, cfg.team_id, only=a.only)
    if a.evaluate:
        _evaluate(a.sequences, a.out, cfg.team_id, a.fps, a.split, a.only)


def _evaluate(
    sequences: str, predictions: str, team_id: int, fps: float, split: str, only: Optional[List[str]] = None
) -> None:
    from soccer_tracker import evaluation

    print(evaluation.format_report(evaluation.evaluate(sequences, predictions, team_id, fps, split, only=only)))


def _cmd_evaluate(a: argparse.Namespace) -> None:
    _evaluate(a.sequences, a.predictions, a.team_id, a.fps, a.split, a.only)


def _cmd_ablate(a: argparse.Namespace) -> None:
    from soccer_tracker import ablation
    from soccer_tracker.io_utils import list_sequences

    rows = ablation.run_ablation(a.sequences, a.raw, a.team_id, a.only, fps=a.fps, split=a.split)
    table = ablation.format_markdown(rows, len(list_sequences(a.sequences, a.only)))
    print(table)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(table + "\n", encoding="utf-8")


def _cmd_render(a: argparse.Namespace) -> None:
    from soccer_tracker import viewer

    print(viewer.render_video(a.frames, a.tracking, a.out, a.roi, a.fps, a.max_frames))


def _cmd_view(a: argparse.Namespace) -> None:
    from soccer_tracker import viewer

    viewer.compare_gt_vs_predictions(a.sequences, a.predictions, a.team_id, a.fps)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="soccer_tracker", description="Soccer player detection, tracking and behaviour analysis.")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(func=func)
        return sp

    s = add("prepare-data", _cmd_prepare_data, "Convert SoccerNet tracking data to a YOLO dataset (optionally download it)")
    s.add_argument("--soccernet-root", default="SoccerNetData")
    s.add_argument("--out", default="YoloDataset")
    s.add_argument("--download", action="store_true", help="download SoccerNet tracking data first (several GB)")
    s.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test", "challenge"], help="splits to download")
    s.add_argument("--symlink", action="store_true", help="symlink instead of copying images")
    s.add_argument("--subsample-step", type=int, default=0, help="also write <split>_subset.txt keeping every N-th image")

    s = add("check-data", _cmd_check_data, "Draw converted YOLO boxes on sample images to validate the conversion")
    s.add_argument("--yolo-root", default="YoloDataset")
    s.add_argument("--split", default="train", choices=["train", "valid"])
    s.add_argument("--num-samples", type=int, default=20)
    s.add_argument("--show", action="store_true", help="open an OpenCV window")
    s.add_argument("--save-dir", default=None)

    s = add("train-detector", _cmd_train_detector, "Fine-tune YOLO11 for person detection")
    s.add_argument("--data", default="YoloDataset/data.yaml")
    s.add_argument("--config", default="configs/yolo_train.yaml")
    s.add_argument("--project", default="runs/detect")
    s.add_argument("--name", default="soccer_tracker")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--device", default="auto")

    s = add("val-detector", _cmd_val_detector, "Precision/recall/mAP of a detector on a YOLO dataset split")
    s.add_argument("--weights", default="weights/yolo11l_soccer.pt")
    s.add_argument("--data", default="YoloDataset/data.yaml")
    s.add_argument("--imgsz", type=int, default=1920)
    s.add_argument("--conf", type=float, default=0.001)
    s.add_argument("--iou", type=float, default=0.65)
    s.add_argument("--split", default="val")
    s.add_argument("--device", default="auto")

    s = add("plot-training", _cmd_plot_training, "Plot an Ultralytics results.csv to PNG")
    s.add_argument("--csv", default="results/detector_training/results.csv")
    s.add_argument("--out", default="results/detector_training/training_curves.png")

    s = add("build-reid-data", _cmd_build_reid, "Crop identity images from SoccerNet GT for ReID training")
    s.add_argument("--soccernet-tracking-root", default="SoccerNetData/tracking")
    s.add_argument("--out", default="reid_dataset")
    s.add_argument("--sample-every", type=int, default=10)
    s.add_argument("--overwrite", action="store_true")

    s = add("train-reid", _cmd_train_reid, "Fine-tune OSNet-AIN with Torchreid (needs torchreid)")
    s.add_argument("--dataset", default="reid_dataset")
    s.add_argument("--pretrained", default="osnet_ain_x1_0_imagenet.pth")
    s.add_argument("--save-dir", default="log/osnet_soccer")
    s.add_argument("--epochs", type=int, default=50)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=3e-4)

    s = add("export-reid", _cmd_export_reid, "Extract a clean state_dict from a Torchreid checkpoint for BoxMOT")
    s.add_argument("checkpoint", help="e.g. log/osnet_soccer/model/model.pth.tar-40")
    s.add_argument("--out", default="weights/osnet_ain_x1_0_soccer.pt")

    s = add("run", _cmd_run, "Full inference pipeline: detect -> track -> post-process -> behaviour (-> evaluate)")
    s.add_argument("--sequences", required=True, help="folder with one sub-folder per video (each with img1/ and roi.json)")
    s.add_argument("--out", default="Predictions_folder")
    s.add_argument("--config", default="configs/pipeline.yaml")
    s.add_argument("--team-id", type=int, default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--no-postprocess", action="store_true")
    s.add_argument("--evaluate", action="store_true", help="also compute HOTA@0.5 / nMAE (needs ground truth)")
    s.add_argument("--only", nargs="+", default=None, metavar="SEQ", help="process only these sequence folder names")
    s.add_argument("--fps", type=float, default=25.0)
    s.add_argument("--split", default="test")

    s = add("evaluate", _cmd_evaluate, "HOTA@0.5, nMAE and PTBS of existing predictions")
    s.add_argument("--sequences", required=True)
    s.add_argument("--predictions", default="Predictions_folder")
    s.add_argument("--team-id", type=int, default=27)
    s.add_argument("--fps", type=float, default=25.0)
    s.add_argument("--split", default="test")
    s.add_argument("--only", nargs="+", default=None, metavar="SEQ", help="evaluate only these sequence folder names")

    s = add("ablate", _cmd_ablate, "Re-score raw tracker output under different post-processing settings")
    s.add_argument("--sequences", required=True)
    s.add_argument("--raw", default="Predictions_folder/raw", help="folder with the raw tracking_<seq>_<team>.txt files")
    s.add_argument("--team-id", type=int, default=27)
    s.add_argument("--only", nargs="+", default=None, metavar="SEQ")
    s.add_argument("--fps", type=float, default=25.0)
    s.add_argument("--split", default="test")
    s.add_argument("--out", default=None, help="also write the markdown table to this file")

    s = add("render", _cmd_render, "Render a tracking file (and ROI counts) to an MP4")
    s.add_argument("--frames", required=True, help="folder with the sequence frames (img1)")
    s.add_argument("--tracking", required=True, help="tracking_<seq>_<team>.txt")
    s.add_argument("--out", default="tracking_output.mp4")
    s.add_argument("--roi", default=None, help="optional roi.json")
    s.add_argument("--fps", type=float, default=25.0)
    s.add_argument("--max-frames", type=int, default=None)

    s = add("view", _cmd_view, "Interactive ground-truth vs prediction viewer")
    s.add_argument("--sequences", required=True)
    s.add_argument("--predictions", default="Predictions_folder")
    s.add_argument("--team-id", type=int, default=27)
    s.add_argument("--fps", type=float, default=25.0)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
