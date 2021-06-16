#
#
#   Base model
#
#

import os
import re
import abc
import mlflow
import tempfile

from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, Union

from ..utils.py_utils import strip_underscores


@dataclass
class TrainerRunInfo:
    info: Any
    metrics: Dict[str, Union[int, float]]
    params: Dict[str, Any]
    tags: Dict[str, Any]


@dataclass
class TrainerInfo:
    run: TrainerRunInfo
    config: DictConfig


class Predictor(metaclass=abc.ABCMeta):

    trainer: TrainerInfo = None

    @classmethod
    def from_pretrained(cls, name_or_run_id, *args, **kwargs):
        source = kwargs.pop("source", "auto").lower()
        force_reload = kwargs.pop("force_reload", False)
        verbose = kwargs.pop("verbose", True)

        if source == "auto":
            if re.match(r"^.+/models/.+$", name_or_run_id):
                source = "sofia"
            else:
                source = "mlflow"

        client = mlflow.tracking.MlflowClient()

        predictor = cls(*args, **kwargs)

        if source == "mlflow":
            run = client.get_run(name_or_run_id)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Download artifacts
                artifacts = client.list_artifacts(name_or_run_id)
                if len(artifacts) > 0:
                    client.download_artifacts(name_or_run_id, "", tmpdir)

                # Load weights
                config = OmegaConf.load(os.path.join(tmpdir, ".hydra", "config.yaml"))
                predictor.load(config.trainer, tmpdir)

                # Add trainer info
                predictor.trainer = TrainerInfo(
                    run=TrainerRunInfo(
                        info=strip_underscores(run.info),
                        metrics=run.data.metrics,
                        params=run.data.params,
                        tags=run.data.tags
                    ),
                    config=config.trainer
                )
        elif source == "sofia":
            ...
            artifacts_dir = None
        else:
            raise ValueError(f'Unknown source: "{source}". Allowed values: "sofia" | "mlflow".')

        return predictor

    def load(self, trainer_config: DictConfig, artifacts_dir: str):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        ...


class Trainer:

    def setup(self, data_dir):
        pass

    @abc.abstractmethod
    def train(self):
        ...

    def test(self):
        raise NotImplementedError(f"Trainer {self.__class__.__name__} does not support test")
