#
#
#   Yolov5 Hydra configuration
#
#

import enum

from dataclasses import dataclass, field
from typing import Union, Optional, List


class OptimizerType(enum.Enum):
    ADAM = "adam"
    SGD = "sgd"


class YOLOArchitecture(enum.Enum):
    YOLOV5S = "yolov5s"
    YOLOV5M = "yolov5m"
    YOLOV5L = "yolov5l"
    YOLOV5X = "yolov5x"


@dataclass
class Yolov5HydraConf:

    classes: List[str]
    num_epochs: int = 300
    batch_size: int = 16
    num_workers: int = 0
    seed: Optional[int] = 0
    gpus: int = -1
    optimizer: OptimizerType = OptimizerType.SGD
    use_linear_lr: bool = False
    img_size: int = 640
    save_best: bool = True  # if true, it will save best, otherwise it will save last
    use_pretrained: bool = False
    freeze: int = 0  # Number of layers to freeze. backbone=10, all=24
    architecture: YOLOArchitecture = YOLOArchitecture.YOLOV5S
    use_sync_bn: bool = False  # Use SyncBatchNorm, only available in DDP mode
    disable_autoanchor_check: bool = False
    label_smoothing: float = 0.0  # Label smoothing epsilon
    use_weighted_images: bool = False  # use weighted image selection for training
    use_multi_scale: bool = False  # vary img-size +/- 50%%
    use_quad: bool = False  # Use quad dataloader
    as_single_cls: bool = False  # train multi-class data as single-class
    use_rect_training: bool = False  # Use rectangular training
    cache: Optional[str] = None  # Cache images in "ram" (default) or "disk"
    only_val_last_epoch: bool = False  # Only validate final epoch
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2

    # Hyperparameters
    lr0: float = 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
    lrf: float = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
    momentum: float = 0.937  # SGD momentum/Adam beta1
    weight_decay: float = 0.0005  # optimizer weight decay 5e-4
    warmup_epochs: float = 3.0  # warmup epochs (fractions ok)
    warmup_momentum: float = 0.8  # warmup initial momentum
    warmup_bias_lr: float = 0.1  # warmup initial bias lr
    box: float = 0.05  # box loss gain
    cls: float = 0.5  # cls loss gain
    cls_pw: float = 1.0  # cls BCELoss positive_weight
    obj: float = 1.0  # obj loss gain (scale with pixels)
    obj_pw: float = 1.0  # obj BCELoss positive_weight
    iou_t: float = 0.20  # IoU training threshold
    anchor_t: float = 4.0  # anchor-multiple threshold
    anchors: Optional[int] = None  # anchors per output layer (0 to ignore)
    fl_gamma: float = 0.0  # focal loss gamma (efficientDet default gamma=1.5)
    hsv_h: float = 0.015  # image HSV-Hue augmentation (fraction)
    hsv_s: float = 0.7  # image HSV-Saturation augmentation (fraction)
    hsv_v: float = 0.4  # image HSV-Value augmentation (fraction)
    degrees: float = 0.0  # image rotation (+/- deg)
    translate: float = 0.1  # image translation (+/- fraction)
    scale: float = 0.5  # image scale (+/- gain)
    shear: float = 0.0  # image shear (+/- deg)
    perspective: float = 0.0  # image perspective (+/- fraction), range 0-0.001
    flipud: float = 0.0  # image flip up-down (probability)
    fliplr: float = 0.5  # image flip left-right (probability)
    mosaic: float = 1.0  # image mosaic (probability)
    mixup: float = 0.0  # image mixup (probability)
    copy_paste: float = 0.0  # segment copy-paste (probability)

    _target_: str = field(init=False, repr=False, default="gymnos.vision.object_detection.yolov5."
                                                          "trainer.Yolov5Trainer")
