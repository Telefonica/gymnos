#
#
#   Predictor
#
#

import os
import torch
import cv2 as cv
import PIL.Image
import numpy as np
import torchvision.transforms.functional as F

from .tool import utils
from .models import Yolov4
from ....base import BasePredictor

from omegaconf import OmegaConf
from collections import namedtuple
from typing import Optional, Union, List
from bounding_box import bounding_box as bb


Yolov4BBox = namedtuple('Yolov4BBox', ["bbox", "bbox_score", "cls_score", "cls_label"])


def to_pil(img: Union[np.ndarray, str, PIL.Image.Image]):
    if isinstance(img, str):
        img = PIL.Image.open(img)
    elif isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(np.uint8(img)).convert("RGB")

    return img


class Yolov4Predictor(BasePredictor):
    """
    TODO: docstring for predictor
    """

    classes: List[str] = None

    def __init__(self, width=None, height=None, confidence_threshold: float = 0.4, nms_threshold: float = 0.6,
                 device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.width = width
        self.height = height

        self.model = None

    def load(self, artifacts_dir):
        config = self.info.trainer.config

        model = Yolov4(None, len(config.classes))
        state_dict = torch.load(os.path.join(artifacts_dir, "checkpoint.pth"), map_location=self.device)
        # I'm not sure why state_dict is prefixed with module but we need to remove it
        state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict)

        model.to(self.device).eval()

        self.model = model

        self.classes = OmegaConf.to_object(config.classes)

        if self.width is None:
            self.width = config.width
        if self.height is None:
            self.height = config.height

    @torch.no_grad()
    def predict(self, img: Union[np.ndarray, str, PIL.Image.Image]):
        """
        Parameters
        ----------
        img

        Returns
        -------
        predictions: List[Yolov4BBox]
            List of bounding boxes with the following properties:
            - ``bbox``: NumPy array in the following order x1, y, x2, y2
            - ``bbox_score``: Bounding box confidence
            - ``cls_score``: Class confidence
            - ``cls_label``: Class label
        """
        if isinstance(img, str):
            img = PIL.Image.open(img)
        elif isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(np.uint8(img)).convert("RGB")

        sized_img = img.resize((self.width, self.height), PIL.Image.ANTIALIAS)

        tensor_imgs = F.to_tensor(sized_img).unsqueeze(0).to(self.device)

        raw_bboxes = self.model(tensor_imgs)

        raw_bboxes = utils.post_processing(raw_bboxes, self.confidence_threshold, self.nms_threshold)[0]

        predictions = []
        for raw_bbox in raw_bboxes:
            bbox = np.array([
                int(raw_bbox[0] * img.width),   # x1
                int(raw_bbox[1] * img.height),  # y1
                int(raw_bbox[2] * img.width),   # x2
                int(raw_bbox[3] * img.height),  # y2
            ])
            predictions.append(Yolov4BBox(bbox, raw_bbox[4], raw_bbox[5], int(raw_bbox[6])))

        return predictions

    def plot(self, img: Union[str, np.ndarray, PIL.Image.Image], predictions: List[Yolov4BBox]):
        pil_img = to_pil(img)
        cv_img = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

        for prediction in predictions:
            label = self.classes[prediction.cls_label]
            bb.add(cv_img, *prediction.bbox, label)

        out_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

        if isinstance(img, PIL.Image.Image):
            return PIL.Image.fromarray(out_img)
        else:
            return out_img
