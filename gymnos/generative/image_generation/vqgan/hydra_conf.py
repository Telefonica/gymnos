#
#
#   Vqgan Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class VqganHydraConf:

    # TODO: define trainer parameters

    _target_: str = field(init=False, repr=False, default="gymnos.generative.image_generation.vqgan."
                                                          "trainer.VqganTrainer")
