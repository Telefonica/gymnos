#
#
#   Predictor
#
#

import os
import torch
import PIL.Image
import cv2 as cv
import numpy as np

from torchvision import transforms
from collections import namedtuple
from typing import Iterable, Union, Optional, List

from .models.yolo import Model
from ....base import BasePredictor
from ....utils.py_utils import lmap
from .hydra_conf import YOLOArchitecture
from .utils.augmentations import letterbox
from .utils.general import non_max_suppression, scale_coords

HERE = os.path.abspath(os.path.dirname(__file__))


def transform(img_size):
    return transforms.Compose([
        transforms.Lambda(lambda img: letterbox(img, img_size)[0]),
        transforms.ToTensor()
    ])


def load_img(img: Union[np.ndarray, PIL.Image.Image, str]):
    if isinstance(img, str):
        img = cv.imread(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    elif isinstance(img, PIL.Image.Image):
        return np.array(img)
    return img


Yolov5BBox = namedtuple("Yolov5BBox", ["bbox", "bbox_score", "cls_score", "cls_label"])


class Yolov5Predictor(BasePredictor):
    """
    TODO: docstring for predictor
    """

    classes: List[str]

    def __init__(self, confidence_threshold: float = 0.25, nms_threshold: float = 0.45, img_size: Optional[int] = None,
                 device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.img_size = img_size
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold

        self._model = None

    def load(self, artifacts_dir):
        config = self.info.trainer.config

        if self.img_size is None:
            self.img_size = config.img_size

        self.classes = config.classes

        architecture = YOLOArchitecture[config.architecture]
        cfg_path = os.path.join(HERE, "models", (architecture.value + ".yaml"))

        nc = 1 if config.as_single_cls else len(config.classes)

        state_dict = torch.load(os.path.join(artifacts_dir, "best.pt"), map_location=self.device)
        state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}

        self._model = Model(cfg=cfg_path, ch=3, nc=nc)
        self._model.load_state_dict(state_dict)
        self._model.eval().to(self.device)

    def predict(self, img: Union[np.ndarray, PIL.Image.Image, str]):
        return self.predict_on_batch([img])[0]

    def predict_on_batch(self, imgs: Iterable[Union[np.ndarray, PIL.Image.Image, str]]):
        imgs = lmap(load_img, imgs)

        img_transform = transform(self.img_size)
        tensor_imgs = torch.stack(lmap(lambda img: img_transform(img).to(self.device), imgs))

        preds = self._model(tensor_imgs)[0]

        results = []
        for pred, tensor_img, original_img in zip(preds, tensor_imgs, imgs):
            pred = non_max_suppression(pred.unsqueeze(0), self.confidence_threshold, self.nms_threshold)[0]

            if pred is None:
                continue

            pred[:, :4] = scale_coords(tensor_img.shape[1:], pred[:, :4], original_img.shape[:2]).round()

            for x1, y1, x2, y2, conf, cls in pred.cpu().numpy():
                print(x1, y1, x2, y2)

            results.append([])

        return results
