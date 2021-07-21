#
#
#   Yolo v5 conf
#
#

from typing import List
from dataclasses import dataclass, field

from typing import Optional


@dataclass
class TransferEfficientNetHydraConf:

    classes: List[str]
    num_workers: int = 0
    batch_size: int = 32
    num_epochs: int = 30
    gpus: int = -1
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    accelerator: Optional[str] = None

    _target_: str = field(init=False, repr=False, default="gymnos.vision.image_classification.transfer_efficientnet."
                                                          "trainer.TransferEfficientNetTrainer")
