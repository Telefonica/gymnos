#
#
#   Trainer
#
#

import os
import mlflow
import torch
import atexit
import shutil
import logging
import inspect
import tempfile
import cv2 as cv
import numpy as np

from tqdm import tqdm
from torch import optim
from dataclasses import dataclass
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

from .loss import YOLOLoss
from .models import Yolov4
from ....base import BaseTrainer
from .dataset import YOLODataset
from .utils import download_pretrained_model
from .tool.tv_reference.coco_eval import CocoEvaluator
from .tool.tv_reference.coco_utils import convert_to_coco_api
from .tool.tv_reference.utils import collate_fn as val_collate
from .hydra_conf import Yolov4HydraConf, Yolov4OptimizerType


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


@torch.no_grad()
def evaluate(model, data_loader, width, height, device):
    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = [[cv.resize(img, (width, height))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        outputs = model(model_input)

        res = {}
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[..., 0] = boxes[..., 0] * img_width
            boxes[..., 1] = boxes[..., 1] * img_height
            boxes[..., 2] = boxes[..., 2] * img_width
            boxes[..., 3] = boxes[..., 3] * img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


@dataclass
class Yolov4Trainer(Yolov4HydraConf, BaseTrainer):
    """
    It expects a ``labels.txt`` file with the following structure:

    .. code-block::

        image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
        image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
        ...
    """

    def __post_init__(self):
        if self.num_workers < 0:
            self.num_workers = cpu_count()

        if self.gpus < 0:
            self.gpus = torch.cuda.device_count()

        if self.mixup is None:
            if self.mosaic and self.cutmix:
                self.mixup = 4
            elif self.cutmix:
                self.mixup = 2
            elif self.mosaic:
                self.mixup = 3
            else:
                raise ValueError("`mosaic` or `cutmix` must be True")

        pretrained_path = None

        if self.use_pretrained:
            pretrained_path = download_pretrained_model()

        self._model = Yolov4(pretrained_path, len(self.classes))

        if self.gpus == 0 or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda")

        if self.gpus > 1:
            self._model = torch.nn.DataParallel(self._model)

        self._model.to(self._device)

    def setup(self, root):
        labels_fpath = os.path.join(root, "labels.txt")
        assert os.path.isfile(labels_fpath), "labels.txt not found"

        with open(labels_fpath) as fp:
            labels = fp.readlines()

        rng = np.random.default_rng(self.seed)

        rng.shuffle(labels)

        train_samples = int(len(labels) * self.train_split)
        val_samples = int(len(labels) * self.val_split)
        test_samples = int(len(labels) * self.test_split)

        test_samples += len(labels) - (train_samples + val_samples + test_samples)

        train_labels, remaining_labels = np.split(labels, [train_samples])
        val_labels, remaining_labels = np.split(remaining_labels, [val_samples])
        test_labels, remaining_labels = np.split(remaining_labels, [test_samples])

        tempfolder = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, tempfolder)

        self._dataset_dir = root
        self._train_labels_fpath = os.path.join(tempfolder, "train.txt")
        self._val_labels_fpath = os.path.join(tempfolder, "val.txt")
        self._test_labels_fpath = os.path.join(tempfolder, "test.txt")

        with open(self._train_labels_fpath, "w") as fp:
            fp.writelines(train_labels)

        with open(self._val_labels_fpath, "w") as fp:
            fp.writelines(val_labels)

        with open(self._test_labels_fpath, "w") as fp:
            fp.writelines(test_labels)

    def train(self):
        logger = logging.getLogger(__name__)

        train_dataset = YOLODataset(self._train_labels_fpath, self._dataset_dir, self.mixup, self.letter_box,
                                    self.width, self.height, self.jitter, self.hue, self.saturation, self.exposure,
                                    self.flip, self.blur, self.gaussian, self.boxes, len(self.classes), train=True)
        val_dataset = YOLODataset(self._val_labels_fpath, self._dataset_dir, self.mixup, self.letter_box, self.width,
                                  self.height, self.jitter, self.hue, self.saturation, self.exposure, self.flip,
                                  self.blur, self.gaussian, self.boxes, len(self.classes), train=False)

        samples_train = len(train_dataset)
        samples_val = len(val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size // self.subdivisions, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True, drop_last=False, collate_fn=collate)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size // self.subdivisions, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True, drop_last=False, collate_fn=val_collate)

        logger.info(inspect.cleandoc(f'''Starting training:
            Epochs:             {self.num_epochs}
            Batch size:         {self.batch_size}
            Subdivisions:       {self.subdivisions}
            Learning rate:      {self.learning_rate}
            Training samples:   {samples_train}
            Validation samples: {samples_val}
            Device:             {self._device.type}
            Images size:        {self.width}
            Optimizer:          {self.optimizer.name}
            Dataset classes:    {self.classes}
            Pretrained:         {self.use_pretrained}
        '''))

        # learning rate setup
        def burnin_schedule(i):
            if i < self.burn_in:
                factor = pow(i / self.burn_in, 4)
            elif i < self.steps[0]:
                factor = 1.0
            elif i < self.steps[1]:
                factor = 0.1
            else:
                factor = 0.01
            return factor

        if self.optimizer == Yolov4OptimizerType.ADAM:
            optimizer = optim.Adam(
                self._model.parameters(),
                lr=self.learning_rate / self.batch_size,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        elif self.optimizer == Yolov4OptimizerType.SGD:
            optimizer = optim.SGD(
                self._model.parameters(),
                lr=self.learning_rate / self.batch_size,
                momentum=self.momentum,
                weight_decay=self.decay,
            )
        else:
            raise ValueError(f"Unexpected optimizer {self.optimizer}")

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

        criterion = YOLOLoss(len(self.classes), batch=self.batch_size // self.subdivisions, device=self._device,
                             n_anchors=self.num_anchors)

        global_step = 0

        for epoch in range(self.num_epochs):
            self._model.train()

            epoch_loss = 0
            epoch_step = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.num_epochs}', leave=False)

            for i, batch in enumerate(pbar):
                epoch_step += 1
                global_step += 1

                images = batch[0]
                bboxes = batch[1]

                images = images.to(self._device, dtype=torch.float32)
                bboxes = bboxes.to(self._device)

                bboxes_pred = self._model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)

                loss.backward()

                epoch_loss += loss.item()

                if (global_step % self.subdivisions) == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (global_step % (self.log_interval_frequency * self.subdivisions)) == 0:
                    mlflow.log_metrics({
                        "train/loss": loss.item(),
                        "train/loss_xy": loss_xy.item(),
                        "train/loss_wh": loss_wh.item(),
                        "train/loss_obj": loss_obj.item(),
                        "train/loss_cls": loss_cls.item(),
                        "train/loss_l2": loss_l2.item(),
                        "lr": scheduler.get_last_lr()[0] * self.batch_size,
                    }, global_step)

            self._model.eval()

            val_evaluator = evaluate(self._model, val_loader, self.width, self.height, self._device)

            stats = val_evaluator.coco_eval['bbox'].stats
            mlflow.log_metrics({
                "val/AP": stats[0],
                "val/AP50": stats[1],
                "val/AP75": stats[2],
                "val/AP_small": stats[3],
                "val/AP_medium": stats[4],
                "val/AP_large": stats[5],
                "val/AR1": stats[6],
                "val/AR10": stats[7],
                "val/AR100": stats[8],
                "val/AR_small": stats[9],
                "val/AR_medium": stats[10],
                "val/AR_large": stats[11],
            }, global_step)

            mlflow.log_metric("epoch", epoch)

        torch.save(self._model.state_dict(), "checkpoint.pth")
        mlflow.log_artifact("checkpoint.pth")

    def test(self):
        test_dataset = YOLODataset(self._test_labels_fpath, self._dataset_dir, self.mixup, self.letter_box, self.width,
                                   self.height, self.jitter, self.hue, self.saturation, self.exposure, self.flip,
                                   self.blur, self.gaussian, self.boxes, len(self.classes), train=False)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size // self.subdivisions, shuffle=True,
                                 num_workers=self.num_workers, pin_memory=True, drop_last=False, collate_fn=val_collate)

        self._model.eval()

        test_evaluator = evaluate(self._model, test_loader, self.width, self.height, self._device)

        stats = test_evaluator.coco_eval['bbox'].stats
        mlflow.log_metrics({
            "test/AP": stats[0],
            "test/AP50": stats[1],
            "test/AP75": stats[2],
            "test/AP_small": stats[3],
            "test/AP_medium": stats[4],
            "test/AP_large": stats[5],
            "test/AR1": stats[6],
            "test/AR10": stats[7],
            "test/AR100": stats[8],
            "test/AR_small": stats[9],
            "test/AR_medium": stats[10],
            "test/AR_large": stats[11],
        })
