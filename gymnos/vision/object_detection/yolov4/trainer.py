#
#
#   Trainer
#
#
import inspect
from dataclasses import dataclass

import mlflow
import atexit
import shutil
import tempfile

from ....base import BaseTrainer
from .hydra_conf import Yolov4HydraConf, Yolov4IouType, Yolov4OptimizerType

import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from multiprocessing import cpu_count

from .dataset import YOLODataset
from .models import Yolov4
from .tool.darknet2pytorch import Darknet

from .tool.tv_reference.utils import collate_fn as val_collate


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class YOLOLoss(nn.Module):

    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super().__init__()

        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


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


@dataclass
class Yolov4Trainer(Yolov4HydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
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

        here = os.path.dirname(os.path.abspath(__file__))

        if self.use_darknet:
            self._model = Darknet(os.path.join(here, "cfg", self.config_file.value))
        else:
            self._model = Yolov4(None, len(self.classes))  # FIXME

        if self.gpus == 0 or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda")

        if self.gpus > 1:
            self._model = torch.nn.DataParallel(self._model)

        self._model.to(self._device)

    def setup(self, root):
        self._dataset_dir = root

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
        val_dataset = YOLODataset(self._val_labels_fpath,  self._dataset_dir, self.mixup, self.letter_box, self.width,
                                  self.height, self.jitter, self.hue, self.saturation, self.exposure, self.flip,
                                  self.blur, self.gaussian, self.boxes, len(self.classes), train=False)

        samples_train = len(train_dataset)
        samples_val = len(val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size // self.subdivisions, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True, drop_last=True, collate_fn=collate)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size // self.subdivisions, shuffle=True,
                                num_workers=self.num_workers, pin_memory=True, drop_last=True, collate_fn=val_collate)

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
            Pretrained:
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

        self._model.train()

        global_step = 0

        for epoch in range(self.num_epochs):
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

                if global_step % (self.log_interval_frequency * self.subdivisions) == 0:
                    mlflow.log_metrics({
                        "train/loss": loss.item(),
                        "train/loss_xy": loss_xy.item(),
                        "train/loss_wh": loss_wh.item(),
                        "train/loss_obj": loss_obj.item(),
                        "train/loss_cls": loss_cls.item(),
                        "train/loss_l2": loss_l2.item(),
                        "lr": scheduler.get_last_lr() * self.batch_size,
                    }, global_step)

                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_last_lr() * self.batch_size
                                        })

            pbar.write(f"Epoch: {epoch}\nStep: {global_step}\nTrain Loss: {epoch_loss}\n")

            mlflow.log_metric("epoch", epoch)

    def test(self):
        test_dataset = YOLODataset(self._test_labels_fpath)
