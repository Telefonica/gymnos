#
#
#   Wgan Gp Hydra configuration
#
#
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class WganGpHydraConf:

    learning_rate: float = 1e-4
    batch_size: int = 64
    image_size: int = 64
    channels_img: int = 3  # RGB
    z_dim: int = 100  # latent size
    num_epochs: int = 100
    features_c: int = 16
    features_g: int = 16
    critic_iterations: int = 5
    lambda_gp: int = 10

    _target_: str = field(init=False, repr=False, default="gymnos.generative.image_generation.wgan_gp."
                                                          "trainer.WganGpTrainer")
