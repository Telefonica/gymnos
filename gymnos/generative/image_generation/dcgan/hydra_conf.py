#
#
#   Dcgan Hydra configuration
#
#

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DCGANHydraConf:

    batch_size: int = 64
    num_epochs: int = 100
    num_workers: int = 0
    latent_size: int = 128
    num_channels: int = 3
    train_split: float = 0.8
    test_split: float = 0.2
    generator_depth: int = 64
    discriminator_depth: int = 64
    log_images_interval: Optional[int] = 5
    generator_learning_rate: float = 1e-3
    discriminator_learning_rate: float = 1e-3

    gpus: int = -1

    _target_: str = field(init=False, repr=False, default="gymnos.generative.image_generation.dcgan."
                                                          "trainer.DCGANTrainer")
