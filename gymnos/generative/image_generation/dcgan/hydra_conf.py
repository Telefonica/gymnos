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
    generator_depth: int = 64  # Size of feature maps in generator
    discriminator_depth: int = 64  # Size of feature maps in discriminator
    log_images_interval: Optional[int] = 5
    generator_learning_rate: float = 2e-4
    discriminator_learning_rate: float = 2e-4
    gpus: int = -1
    beta1: float = 0.5  # Beta1 hyperparam for Adam optimizers

    _target_: str = field(init=False, repr=False, default="gymnos.generative.image_generation.dcgan."
                                                          "trainer.DCGANTrainer")
