#
#
#   Data module
#
#

import os
import logging
import pytorch_lightning as pl
import torchvision.transforms as T

from typing import Optional
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from fastdl.extractor import extract_file

from .utils import split_indices


class TransferEfficientNetDataModule(pl.LightningDataModule):

    def __init__(self, root: str, classes, split=(0.6, 0.2, 0.2), num_workers: int = 0, batch_size: int = 32,
                 random_state: Optional[int] = None):
        super().__init__()

        if random_state is None:
            random_state = 0

        self.root = root
        self.classes = classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.split = split
        self.random_state = random_state

        self._train_indices, self._val_indices, self._test_indices = None, None, None

    def prepare_data(self):
        logger = logging.getLogger(__name__)

        logger.info("Extracting files ...")
        for classname in self.classes:
            extract_file(os.path.join(self.root, classname + ".zip"), extract_dir=os.path.join(self.root, classname))

    def setup(self, stage=None):
        dataset = ImageFolder(self.root)
        self._train_indices, self._val_indices, self._test_indices = split_indices(len(dataset), self.split,
                                                                                   shuffle=True,
                                                                                   random_state=self.random_state)

    def train_dataloader(self):
        transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomRotation(20, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(self.root, transform)
        dataset = Subset(dataset, self._train_indices)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(self.root, transform)
        dataset = Subset(dataset, self._val_indices)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(self.root, transform)
        dataset = Subset(dataset, self._test_indices)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
