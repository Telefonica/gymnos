#
#
# Efficient-net image classifier
#
#

from .trainer import TransferEfficientNetTrainer
from .predictor import TransferEfficientNetPredictor


dependencies = [
    "numpy",
    "torch",
    "Pillow",
    "torchvision",
    "torchmetrics",
    "efficientnet_pytorch==0.7.0",
    "pytorch-lightning>=1.0.0"
]
