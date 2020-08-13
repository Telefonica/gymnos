#
#
#   wondervision
#
#

import os
import uuid

from datetime import datetime

from ..utils import lazy_imports as lazy
from ..utils.json_utils import save_to_json
from .execution_environment import ExecutionEnvironment


class _GymnosLogger(lazy.pytorch_lightning.loggers.LightningLoggerBase):

    def __init__(self, logger, run_id=None):
        super().__init__()

        self.run_id = run_id
        self.logger = logger

    @property
    def experiment(self):
        return self.logger

    @property
    def name(self):
        return "gymnos"

    @property
    def version(self):
        return self.run_id

    @lazy.pytorch_lightning.utilities.rank_zero_only
    def log_hyperparams(self, params):
        self.logger.log_params(vars(params))

    @lazy.pytorch_lightning.utilities.rank_zero_only
    def log_metrics(self, metrics, step):
        self.logger.log_metrics(metrics, step=step)

    @lazy.pytorch_lightning.utilities.rank_zero_only
    def finalize(self, status):
        self.logger.end()


class WonderVision(ExecutionEnvironment):
    """
    Execution environment to run experiments using `wondervision <https://github.com/Telefonica/wondervision/>`_.
    framework.
    Check ``examples/experiments/wondervision.json`` to see a wondervision experiment.

    Parameters
    ------------
    config_files: list of str, optional
        List of JSON paths to look for configuration values.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--execution_dir", help="Execution directory to store training outputs. It accepts the " +
                                                    "following format arguments: dataset_type, model_type, uuid, now",
                            type=str, default="trainings/wondervision/{now:%Y-%m-%d_%H-%M-%S}")
        parser.add_argument("--trackings_dir", help="Execution directory to store tracking outputs. It accepts the" +
                                                    " following format arguments: now, uuid",
                            type=str, default="trainings/wondervision/{now:%Y-%m-%d_%H-%M-%S}")

    def train(self, trainer, **kwargs):
        format_kwargs = dict(
            uuid=uuid.uuid4().hex,
            now=datetime.now()
        )

        execution_dir = kwargs["execution_dir"].format(**format_kwargs)
        trackings_dir = kwargs["trackings_dir"].format(**format_kwargs)

        os.makedirs(execution_dir)
        os.makedirs(trackings_dir, exist_ok=True)

        trainer.tracking.trackers.start(trainer.tracking.run_id, trackings_dir)

        pl_trainer = lazy.pytorch_lightning.Trainer(
            logger=_GymnosLogger(trainer.tracking.trackers, trainer.tracking.run_id),
            weights_save_path=execution_dir,
            **trainer.model.training)

        pl_trainer.fit(trainer.model.model)

        pl_trainer.test(trainer.model.model)

        pl_trainer.save_checkpoint(os.path.join(execution_dir, "model.ckpt"))

        save_to_json(os.path.join(execution_dir, "experiment.json"), trainer.to_dict())

        return pl_trainer
