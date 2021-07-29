#
#
#   Model
#
#

from .hydra_conf import Yolov4HydraConf

hydra_conf = Yolov4HydraConf

dependencies = [
    "torch",
    "numpy",
    "torchvision",
    "opencv-python",
    "Cython",
    "pycocotools>=2.0.2"
]
