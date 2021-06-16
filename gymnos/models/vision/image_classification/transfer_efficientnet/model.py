#
#
#   Model
#
#

from torchmetrics.classification import Accuracy

import torch
import torch.nn as nn
import efficientnet_pytorch
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F


class TransferEfficientNetModule(pl.LightningModule):

    def __init__(self, num_classes):
        super().__init__()

        self.base = efficientnet_pytorch.EfficientNet.from_pretrained(
            "efficientnet-b0",
            in_channels=3,
            include_top=False,
            advprop=False
        )

        for name, param in self.base.named_parameters():
            if "_bn" not in name:
                param.requires_grad = False

        # Freeze all layers except the last one
        for param in self.base._conv_head.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.train_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        logits = self(batch[0])
        loss = F.cross_entropy(logits, batch[1])

        self.log("train_loss", loss)

        self.train_accuracy(torch.softmax(logits, 1), batch[1])

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch[0])
        loss = F.cross_entropy(logits, batch[1])

        self.val_accuracy(torch.softmax(logits, 1), batch[1])

        return {
            "loss": loss
        }

    def test_step(self, batch, batch_idx):
        logits = self(batch[0])
        self.test_accuracy(torch.softmax(logits, 1), batch[1])

    def training_epoch_end(self, outputs):
        self.log("train_acc", self.train_accuracy.compute())

    def validation_epoch_end(self, outputs):
        losses = torch.stack([x["loss"] for x in outputs])

        self.log("val_acc", self.val_accuracy.compute())
        self.log("val_loss", losses.mean())

    def test_epoch_end(self, outputs):
        self.log("test_acc", self.test_accuracy.compute())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
