#
#
#   Forecaaster Autoreg Multi Output Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class ForecaasterAutoregMultiOutputHydraConf:

    # TODO: define trainer parameters

    _target_: str = field(init=False, repr=False, default="gymnos.tabular.regression.forecaaster_autoreg_multi_output."
                                                          "trainer.ForecaasterAutoregMultiOutputTrainer")
