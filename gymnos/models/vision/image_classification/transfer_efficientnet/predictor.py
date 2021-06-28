#
#
#   Predictor
#
#

import os
import PIL
import glob
import torch
import warnings
import numpy as np
import torchvision.transforms as T

from PIL import Image
from typing import Union
from dataclasses import dataclass

from ....base import Predictor
from .model import TransferEfficientNetModule


@dataclass
class TransferEfficientNetPrediction:

    label: int
    probabilities: np.ndarray


class TransferEfficientNetPredictor(Predictor):

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.model = None
        self.classes = None

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load(self, artifacts_dir):
        checkpoints = glob.glob(os.path.join(artifacts_dir, "*.ckpt"))
        if len(checkpoints) == 0:
            raise ValueError("No checkpoint found")
        if len(checkpoints) > 1:
            warnings.warn("More than one checkpoint found. Selecting the first one")

        self.model = TransferEfficientNetModule.load_from_checkpoint(checkpoints[0],
                                                                     num_classes=len(self.trainer.config.classes))
        self.classes = sorted(self.trainer.config.classes)

        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, PIL.Image.Image, str]):
        if isinstance(image, str):
            image = PIL.Image.open(image)
        elif isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(np.uint8(image)).convert("RGB")

        img_tensor = self.transform(image)
        img_tensor = img_tensor.to(self.device)

        logits = self.model(torch.unsqueeze(img_tensor, 0))
        probabilities = torch.softmax(logits, 1)

        class_prediction = torch.argmax(probabilities, 1)

        return TransferEfficientNetPrediction(
            label=class_prediction.item(),
            probabilities=probabilities.cpu().numpy()[0]
        )
