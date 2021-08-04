#
#
#   Trainer
#
#

import os
import time
import math

import torch
import atexit
import random
import shutil
import fastdl
import tempfile
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional
import torch.distributed as dist

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.cuda import amp
from dataclasses import dataclass, asdict
from torch.optim import Adam, SGD, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from .models.yolo import Model
from ....base import BaseTrainer
from .utils.loss import ComputeLoss
from ....config import get_gymnos_home
from .utils.autoanchor import check_anchors
from .utils.datasets import create_dataloader
from .hydra_conf import Yolov5HydraConf, YOLOArchitecture, OptimizerType
from .utils.torch_utils import ModelEMA, select_device, intersect_dicts
from .utils.general import (labels_to_class_weights, labels_to_image_weights, init_seeds,
                            check_img_size, one_cycle, colorstr)

FILE = Path(__file__).absolute()
DIR = Path(__file__).absolute().parent
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def download_pretrained(architecture: YOLOArchitecture) -> str:
    return fastdl.download(
        url=f"http://obiwan.hi.inet/public/gymnos/yolov5/{architecture.value + '.pt'}",
        dir_prefix=os.path.join(get_gymnos_home(), "downloads", "yolov5")
    )


@dataclass
class Yolov5Trainer(Yolov5HydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def __post_init__(self):
        assert self.train_split + self.val_split + self.test_split == 1.0

        if isinstance(self.gpus, int):
            if not torch.cuda.is_available() or self.gpus == 0:
                yolo_device = "cpu"
            elif self.gpus < 0:
                yolo_device = ",".join(range(torch.cuda.device_count()))
            else:
                yolo_device = ",".join(range(self.gpus))
        else:
            yolo_device = ",".join([str(gpu) for gpu in self.gpus])

        device = select_device(yolo_device, batch_size=self.batch_size)

        if LOCAL_RANK != -1:
            from datetime import timedelta
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            assert self.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
            assert not self.use_weighted_images, '--image-weights argument is not compatible with DDP training'
            # assert not opt.evolve, '--evolve argument is not compatible with DDP training'
            assert not self.use_sync_bn, 'known training issue, see https://github.com/ultralytics/yolov5/issues/3998'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                    timeout=timedelta(seconds=60))

        self._device = device

    def setup(self, root):
        tempfolder = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, tempfolder)

        rng = np.random.default_rng(self.seed)

        train_images_dir = os.path.join(tempfolder, "train", "images")
        train_labels_dir = os.path.join(tempfolder, "train", "labels")
        val_images_dir = os.path.join(tempfolder, "val", "images")
        val_labels_dir = os.path.join(tempfolder, "val", "labels")
        test_images_dir = os.path.join(tempfolder, "test", "images")
        test_labels_dir = os.path.join(tempfolder, "test", "labels")

        os.makedirs(train_images_dir)
        os.makedirs(train_labels_dir)
        os.makedirs(val_images_dir)
        os.makedirs(val_labels_dir)
        os.makedirs(test_images_dir)
        os.makedirs(test_labels_dir)

        with open(os.path.join(root, "labels.txt")) as fp_labels:
            for i, line in enumerate(fp_labels):
                random_num = rng.random()

                if 0 <= random_num < self.train_split:
                    images_dir, labels_dir = train_images_dir, train_labels_dir
                elif self.train_split <= random_num < (self.train_split + self.val_split):
                    images_dir, labels_dir = val_images_dir, val_labels_dir
                else:
                    images_dir, labels_dir = test_images_dir, test_labels_dir

                img_path, *bboxes = line.split(" ")

                src_img_path, dst_img_path = os.path.join(root, img_path), os.path.join(images_dir, img_path)

                os.symlink(src_img_path, dst_img_path)

                fname, extension = os.path.splitext(img_path)
                label_path = os.path.join(labels_dir, fname + ".txt")

                with open(label_path, "w") as fp_img_labels:
                    for bbox in bboxes:
                        x1, y1, x2, y2, cls_id = map(int, bbox.split(","))
                        x_center, y_center = (x2 - x1) / 2, (y2 - y1) / 2
                        width, height = (x2 - x1), (y2 - y1)
                        img = Image.open(src_img_path)
                        fp_img_labels.write("{} {} {} {} {}\n".format(cls_id, x_center / img.width,
                                                                      y_center / img.height,
                                                                      width / img.width, height / img.height))

        self._train_images_path = train_images_dir
        self._val_images_path = val_images_dir
        self._test_images_path = test_images_dir

    def train(self):
        logger = logging.getLogger(__name__)

        init_seeds(1 + RANK)

        nc = 1 if self.as_single_cls else len(self.classes)
        is_cuda = self._device.type != "cpu"

        cfg_path = DIR / "models" / (self.architecture.value + ".yaml")

        if self.use_pretrained:
            weights_path = download_pretrained(self.architecture)
            ckpt = torch.load(weights_path, map_location=self._device)
            model = Model(cfg_path, ch=3, nc=nc, anchors=self.anchors)
            exclude = ['anchor']
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            logger.info(f'Transferred {len(csd)}/{len(model.state_dict())} items')  # report
        else:
            model = Model(cfg_path, ch=3, nc=nc, anchors=self.anchors)

        model.to(self._device)

        # Freeze
        freeze = [f'model.{x}.' for x in range(self.freeze)]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                v.requires_grad = False

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / self.batch_size), 1)  # accumulate loss before optimizing
        self.weight_decay *= self.batch_size * accumulate / nbs  # scale weight_decay
        logger.info(f"Scaled weight_decay = {self.weight_decay}")

        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        if self.optimizer == OptimizerType.ADAM:
            optimizer = Adam(g0, lr=self.lr0, betas=(self.momentum, 0.999))  # adjust beta1 to momentum
        else:
            optimizer = SGD(g0, lr=self.lr0, momentum=self.momentum, nesterov=True)

        optimizer.add_param_group({'params': g1, 'weight_decay': self.weight_decay})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)

        del g0, g1, g2

        # Scheduler
        if self.use_linear_lr:
            lf = lambda x: (1 - x / (self.num_epochs - 1)) * (1.0 - self.lrf) + self.lrf  # linear
        else:
            lf = one_cycle(1, self.lrf, self.num_epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        ema = ModelEMA(model) if RANK in [-1, 0] else None

        # Image sizes
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz = check_img_size(self.img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # DP mode
        if is_cuda and RANK == -1 and torch.cuda.device_count() > 1:
            logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                            'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get '
                            'started.')
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if self.use_sync_bn and is_cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self._device)
            logger.info('Using SyncBatchNorm()')

        # Trainloader
        train_loader, dataset = create_dataloader(self._train_images_path, imgsz, self.batch_size // WORLD_SIZE, gs,
                                                  self.as_single_cls, hyp=asdict(self), augment=True, cache=self.cache,
                                                  rect=self.use_rect_training, rank=RANK,
                                                  workers=self.num_workers, image_weights=self.use_weighted_images,
                                                  quad=self.use_quad, prefix=colorstr('train: '))
        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(train_loader)  # number of batches
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in [-1, 0]:
            val_loader = create_dataloader(self._val_images_path, imgsz, self.batch_size // WORLD_SIZE * 2, gs,
                                           self.as_single_cls, hyp=asdict(self), rank=-1, workers=self.num_workers,
                                           cache=None if self.only_val_last_epoch else self.cache, rect=True,
                                           pad=0.5, prefix=colorstr('val: '))[0]

            # Anchors
            if not self.disable_autoanchor_check:
                check_anchors(dataset, model=model, thr=self.anchor_t, imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        # DDP mode
        if is_cuda and RANK != -1:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

        # Model parameters
        self.box *= 3. / nl  # scale to layers
        self.cls *= nc / 80. * 3. / nl  # scale to classes and layers
        self.obj *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        model.nc = nc  # attach number of classes to model
        model.hyp = asdict(self)  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(self._device) * nc  # attach class weights
        model.names = self.classes

        # Start training
        t0 = time.time()
        start_epoch, best_fitness = 0, 0.0
        nw = max(round(self.warmup_epochs * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=is_cuda)
        compute_loss = ComputeLoss(model)  # init loss class
        logger.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {train_loader.num_workers} dataloader workers\n'
                    f'Starting training for {self.num_epochs} epochs...')

        for epoch in range(start_epoch, self.num_epochs):
            model.train()

            # Update image weights (optional)
            if self.use_weighted_images:
                # Generate indices
                if RANK in [-1, 0]:
                    cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                    iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                    dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
                # Broadcast if DDP
                if RANK != -1:
                    indices = (torch.tensor(dataset.indices) if RANK == 0 else torch.zeros(dataset.n)).int()
                    dist.broadcast(indices, 0)
                    if RANK != 0:
                        dataset.indices = indices.cpu().numpy()

            mloss = torch.zeros(3, device=self._device)  # mean losses

            if RANK != -1:
                train_loader.sampler.set_epoch(epoch)

            logger.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))

            pbar = enumerate(train_loader)
            if RANK in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar

            optimizer.zero_grad()

            for i, (imgs, targets, paths, _) in pbar:
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self._device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / self.batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [self.warmup_bias_lr if j == 2 else 0.0,
                                                     x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.warmup_momentum, self.momentum])

                # Multi-scale
                if self.use_multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        # new shape (stretched to gs-multiple)
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=is_cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(self._device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if self.use_quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni - last_opt_step >= accumulate:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    last_opt_step = ni

                # Log
                if RANK in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                        f'{epoch}/{self.num_epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            scheduler.step()

    def test(self):
        pass   # OPTIONAL: test code
