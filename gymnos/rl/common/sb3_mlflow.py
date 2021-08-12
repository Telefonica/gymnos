#
#
#   Mlflow integration for Stable-baselines 3
#
#

import mlflow

from typing import Any, Dict, Union, Tuple
from stable_baselines3.common.logger import KVWriter


class MlflowKVWriter(KVWriter):

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0) -> None:
        mlflow.log_metrics(key_values, step=step)

    def close(self):
        pass
