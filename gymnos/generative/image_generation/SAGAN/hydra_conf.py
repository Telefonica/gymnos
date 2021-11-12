#
#
#   S A G A N Hydra configuration
#
#
from typing import Optional
from dataclasses import dataclass, field

from numpy import string_


@dataclass
class SAGANHydraConf:

    # Model hyper-parameters
    batch_size: int = 64
    imsize: int = 64
    g_num: int = 5
    z_dim: int = 128
    g_conv_dim: int = 64
    d_conv_dim: int = 64
    parallel: bool = False
    adv_loss: str = 'hinge'

    lambda_gp: int = 10
    total_step: int = 1000000
    d_iters: int = 5
    num_workers: int = 2
    g_lr: float = 0.0001
    d_lr: float = 0.0004
    lr_decay: float = 0.95
    beta1: float = 0.0
    beta2: float = 0.9
    pretrained_model = None
    latent_size: int = 128

    log_images_interval: Optional[int] = 5
    dataset: str = 'celeb'
    use_tensorboard = False
    image_path = './data'
    log_path = './logs'
    model_save_path = './models'
    sample_path = './samples'
    log_step: int = 10
    sample_step: int = 10
    model_save_step: float = 1.0
    version = 'sagan_celeb'



    _target_: str = field(init=False, repr=False, default="gymnos.generative.image_generation.SAGAN."
                                                          "trainer.SAGANTrainer")
