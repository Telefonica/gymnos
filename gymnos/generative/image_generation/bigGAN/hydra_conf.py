#
#
#   Big G A N Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class BigGANHydraConf:

    # TODO: define trainer parameters

    _target_: str = field(init=False, repr=False, default="gymnos.generative.image_generation.bigGAN."
                                                          "trainer.BigGANTrainer")
