#
#
#   Model
#
#

from .conf import TransferEfficientNetConf

name = "vision.image_classification.transfer_efficientnet"

conf = TransferEfficientNetConf

dependencies = [
    "numpy",
    "torch",
    "Pillow",
    "torchvision",
    "torchmetrics",
    "efficientnet_pytorch==0.7.0",
    "pytorch-lightning>=1.0.0"
]
