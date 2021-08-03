#
#
#   Yolov4 Hydra configuration
#
#

import enum

from typing import List, Optional
from dataclasses import dataclass, field


class Yolov4OptimizerType(enum.Enum):
    ADAM = "adam"
    SGD = "sgd"


@dataclass
class Yolov4HydraConf:

    classes: List[str]
    batch_size: int = 64
    subdivisions: int = 16
    num_epochs: int = 300
    learning_rate: float = 0.001
    gpus: int = -1
    num_workers: int = 0
    optimizer: Yolov4OptimizerType = Yolov4OptimizerType.ADAM
    log_interval_frequency: int = 20
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    seed: int = 0
    momentum: float = 0.949
    decay: float = 0.0005
    burn_in: int = 1000
    steps: List[int] = (400000, 450000)
    width: int = 608
    height: int = 608
    mixup: Optional[int] = None
    letter_box: bool = False
    jitter: float = 0.2
    hue: float = 0.1
    saturation: float = 1.5
    exposure: float = 1.5
    flip: bool = True
    blur: bool = False
    gaussian: bool = False
    boxes: int = 60
    num_anchors: int = 3
    cutmix: bool = False
    mosaic: bool = True
    use_pretrained: bool = False

    _target_: str = field(init=False, repr=False, default="gymnos.vision.object_detection.yolov4."
                                                          "trainer.Yolov4Trainer")
