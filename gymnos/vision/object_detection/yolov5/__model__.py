#
#
#   Model
#
#

from .hydra_conf import Yolov5HydraConf

hydra_conf = Yolov5HydraConf

pip_dependencies = [
    "matplotlib>=3.2.2",
    "numpy>=1.18.5",
    "opencv-python>=4.1.2",
    "Pillow",
    "PyYAML>=5.3.1",
    "scipy>=1.4.1",
    "torch>=1.7.0",
    "torchvision>=0.8.1",
    "tqdm>=4.41.0"
]

apt_dependencies = []
