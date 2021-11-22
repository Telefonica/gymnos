#
#
#   Trainer
#
#
import os
import mlflow
import logging

from glob import glob
from tqdm import tqdm, trange
import multiprocessing as mp
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as T
from torchvision.utils import save_image

from .utils import *

from ....base import BaseTrainer
from .dataset import SAGANDataset
from .hydra_conf import SAGANHydraConf
from .networks import Generator, Discriminator


def build_transform():
    return T.Compose([
        T.Resize(SAGANHydraConf.imsize),
        T.CenterCrop(SAGANHydraConf.imsize),
        T.ToTensor(),
        T.Lambda(lambda img: (img * 2) - 1)  # normalize values between -1 and 1
    ])


@dataclass
class SAGANTrainer(SAGANHydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def __post_init__(self):
        if self.num_workers < 0:
            self.num_workers = mp.cpu_count()
        if self.gpus < 0:
            self.gpus = torch.cuda.device_count()

        assert self.total_step > 0

    def prepare_data(self, root):
        logger = logging.getLogger(__name__)

        self._filepaths = (glob(os.path.join(root, "*.png")) + glob(os.path.join(root, "*.jpg")) +
                           glob(os.path.join(root, "*.gif")))
        logger.info(f"Found {len(self._filepaths)} images")

    def train(self):

        # Data loader
        transform = build_transform()
        dataset = SAGANDataset(self._filepaths, transform)
        data_loader = DataLoader(dataset, self.batch_size, num_workers=self.num_workers)

        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        for step in range(start, self.total_step):
            running_metrics = {}
            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            for real_images in tqdm(data_loader):

                # Compute loss with real images
                # dr1, dr2, df1, df2, gf1, gf2 are attention scores
                real_images = tensor2var(real_images)
                d_out_real, dr1, dr2 = self.D(real_images)
                if self.adv_loss == 'wgan-gp':
                    d_loss_real = - torch.mean(d_out_real)
                elif self.adv_loss == 'hinge':
                    d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

                # apply Gumbel Softmax
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
                fake_images, gf1, gf2 = self.G(z)
                d_out_fake, df1, df2 = self.D(fake_images)

                if self.adv_loss == 'wgan-gp':
                    d_loss_fake = d_out_fake.mean()
                elif self.adv_loss == 'hinge':
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                if self.adv_loss == 'wgan-gp':
                    # Compute gradient penalty
                    alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                    interpolated = Variable(alpha * real_images.data + (1 - alpha) *
                                            fake_images.data, requires_grad=True)
                    out, _, _ = self.D(interpolated)

                    grad = torch.autograd.grad(outputs=out,
                                               inputs=interpolated,
                                               grad_outputs=torch.ones(out.size()).cuda(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                    # Backward + Optimize
                    d_loss = self.lambda_gp * d_loss_gp

                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                # ================== Train G and gumbel ================== #
                # Create random noise
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
                fake_images, _, _ = self.G(z)

                # Compute loss with fake images
                g_out_fake, _, _ = self.D(fake_images)  # batch x n
                g_loss_fake = - g_out_fake.mean()

                self.reset_grad()
                g_loss_fake.backward()
                self.g_optimizer.step()

                log_metrics = [
                    ("train/discriminator_loss", d_loss.item()),
                    ("train/generator_loss", g_loss_fake.item())
                ]
                for metric_name, metric_value in log_metrics:
                    running_metrics[metric_name] = running_metrics.get(metric_name, 0) + metric_value

            mean_epoch_metrics = {k: v / len(data_loader) for k, v in running_metrics.items()}
            mlflow.log_metrics({**mean_epoch_metrics, "epoch": step})

            if (step == self.total_step - 1) or (self.log_images_interval is not None and
                                                 (step % self.log_images_interval) == 0):
                with torch.no_grad():
                    fake_imgs, _, _ = self.G(self.noise)

                fake_imgs = (fake_imgs + 1) / 2  # denormalize

                val_img = to_pil_image(make_grid(fake_imgs, 5)).convert("L" if self.num_channels == 1 else "RGB")
                mlflow.log_image(val_img, f"epoch-{step}.png")

    def build_model(self):
        device = "cuda" if (torch.cuda.is_available() and self.gpus > 0) else "cpu"
        self.noise = torch.randn(10, self.latent_size, 1, 1, device=device)
        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim).to(device)
        self.D = Discriminator(self.batch_size, self.imsize, self.d_conv_dim).to(device)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
