#
#
#   Trainer
#
#

import mlflow
import torch
import pytorch_lightning as pl

from dataclasses import dataclass
from multiprocessing import cpu_count

from .utils import MLFlowLogger
from ....base import BaseTrainer
from .module import TransferEfficientNetModule
from .hydra_conf import TransferEfficientNetHydraConf
from .datamodule import TransferEfficientNetDataModule


@dataclass
class TransferEfficientNetTrainer(TransferEfficientNetHydraConf, BaseTrainer):
    """
    The expected structure is one directory for each class.

    Let's say we have ``classes=["dog", "cat"]``, then we should have two directories ``"dog"`` and ``"cat"``
    containing the images:

    .. code-block::

        dog/
            img1.png
            img2.png
            ...
        cat/
            img1.png
            img2.png
            ...

    Parameters
    ----------
    classes:
        Classes to train on, each class should be the name of the folder
    num_workers:
        Num workers to load data. If ``0``, loading data will be synchronous. If ``-1``, all CPUs will be used.
    batch_size:
        Batch size for training and testing
    num_epochs:
        Number of epochs to train
    """

    def __post_init__(self):
        if self.gpus < 0:
            self.gpus = torch.cuda.device_count()

        if self.num_workers < 0:
            self.num_workers = cpu_count()

        if self.gpus > 0:
            self.accelerator = "dp"

        self.model = TransferEfficientNetModule(len(self.classes))

        self.trainer = pl.Trainer(
            max_epochs=self.num_epochs,
            gpus=self.gpus,
            accelerator=self.accelerator,
            logger=MLFlowLogger(tracking_uri=mlflow.get_tracking_uri()),
            callbacks=[
                pl.callbacks.ModelCheckpoint("checkpoints", monitor="val_loss")
            ]
        )

    def prepare_data(self, data_dir):
        self.datamodule = TransferEfficientNetDataModule(data_dir, self.classes,
                                                         (self.train_split, self.val_split, self.test_split),
                                                         self.num_workers, self.batch_size)

    def train(self):
        try:
            self.trainer.fit(self.model, datamodule=self.datamodule)
        finally:
            if self.trainer.checkpoint_callback.best_model_path:
                mlflow.log_artifact(self.trainer.checkpoint_callback.best_model_path)

    def test(self):
        self.trainer.test(datamodule=self.datamodule)
