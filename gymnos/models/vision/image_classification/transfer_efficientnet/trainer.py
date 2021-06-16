#
#
#   Trainer
#
#

import torch
import pytorch_lightning as pl

from multiprocessing import cpu_count

from ....base import Trainer
from .model import TransferEfficientNetModule
from .datamodule import TransferEfficientNetDataModule
from .utils import get_lightning_mlflow_logger, MlflowModelLoggerArtifact


class TransferEfficientNetTrainer(Trainer):

    def __init__(self, classes, num_workers: int = 0, batch_size: int = 32, num_epochs: int = 30, gpus: int = -1,
                 train_split: float = 0.6, val_split: float = 0.2, test_split: float = 0.2):
        if gpus < 0:
            gpus = torch.cuda.device_count()
        if num_workers < 0:
            num_workers = cpu_count()

        self.classes = classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gpus = gpus
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.datamodule = None
        self.model = TransferEfficientNetModule(len(classes))

        self.trainer = pl.Trainer(
            max_epochs=num_epochs,
            gpus=self.gpus,
            logger=get_lightning_mlflow_logger(),
            callbacks=[
                MlflowModelLoggerArtifact()
            ]
        )

    def setup(self, data_dir):
        self.datamodule = TransferEfficientNetDataModule(data_dir, self.classes,
                                                         (self.train_split, self.val_split, self.test_split),
                                                         self.num_workers, self.batch_size)

    def train(self):
        self.trainer.fit(self.model, datamodule=self.datamodule)

    def test(self):
        self.trainer.test(datamodule=self.datamodule)
