#
#
#   Calories Intake Hydra conf
#
#

from dataclasses import dataclass, field


@dataclass
class CaloriesIntakeHydraConf:

    # TODO: add custom parameters

    _target_: str = field(init=False, repr=False, default="gymnos.datasets.calories_intake.dataset.CaloriesIntake")
