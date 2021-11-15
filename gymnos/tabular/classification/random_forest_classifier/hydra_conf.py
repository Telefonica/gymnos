#
#
#   Random Forest Classifier Hydra configuration
#
#

from dataclasses import dataclass, field


@dataclass
class RandomForestClassifierHydraConf:

    # TODO: define trainer parameters

    _target_: str = field(init=False, repr=False, default="gymnos.tabular.classification.random_forest_classifier."
                                                          "trainer.RandomForestClassifierTrainer")
