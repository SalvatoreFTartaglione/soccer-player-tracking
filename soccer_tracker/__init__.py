"""Soccer Tracker: player detection, tracking and behaviour analysis for broadcast soccer video.

Pipeline: YOLO11 detector -> BoT-SORT + fine-tuned OSNet-AIN ReID -> tracklet post-processing
-> ROI player counting. Heavy dependencies (torch, ultralytics, boxmot, torchreid) are imported
lazily so that the pure-Python parts (post-processing, behaviour analysis, I/O) can be used and
tested without a GPU stack.
"""

__version__ = "1.0.0"
