"""
Implementation for YOLOv4

Paper: `arxiv.org/abs/2004.10934 <https://arxiv.org/abs/2004.10934>`_
"""

from ....utils import lazy_import

# Public API
Yolov4Predictor = lazy_import("gymnos.vision.object_detection.yolov4.predictor.Yolov4Predictor")
