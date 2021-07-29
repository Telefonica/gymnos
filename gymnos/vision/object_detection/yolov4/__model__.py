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
    "torchvision",
    "opencv-python",
    "Cython",
    "pycocotools>=2.0.2"
]

apt_dependencies = [
    "libgl1-mesa-glx"
]
