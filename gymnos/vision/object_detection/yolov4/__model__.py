#
#
#   Model
#
#

from .hydra_conf import Yolov4HydraConf

hydra_conf = Yolov4HydraConf

pip_dependencies = [
    "torch",
    "numpy",
    "Pillow",
    "Cython",
    "torchvision",
    "bounding-box",
    "opencv-python",
    "pycocotools>=2.0.2",
]

apt_dependencies = [
    "libgl1-mesa-glx"
]
