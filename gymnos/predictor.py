#
#
#   Predictor
#
#

import os
import re
import abc
import mlflow
import tempfile

from dacite import from_dict
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, Union

from .services.sofia import SOFIA
from .utils.mlflow_utils import jsonify_mlflow_run


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

                # Add trainer info
                predictor.trainer = TrainerInfo(
                    run=from_dict(TrainerRunInfo, jsonify_mlflow_run(run)),
                    config=config.trainer
                )

                predictor.load(tmpdir)
        elif source == "sofia":
            response = SOFIA.get_model(name_or_run_id)
            response.raise_for_status()

            data = response.json()

            artifacts_dir = SOFIA.download_model_artifacts(name_or_run_id, force_download=force_reload,
                                                           force_extraction=force_reload, verbose=verbose)

            config = OmegaConf.load(os.path.join(artifacts_dir, ".hydra", "config.yaml"))

            predictor.trainer = TrainerInfo(
                run=from_dict(TrainerRunInfo, data["run"]),
                config=config.trainer
            )

            predictor.load(artifacts_dir)
        else:
            raise ValueError(f'Unknown source: "{source}". Allowed values: "sofia" | "mlflow"')

        return predictor

    def load(self, artifacts_dir: str):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        ...


