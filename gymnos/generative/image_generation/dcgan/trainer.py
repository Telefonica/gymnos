#
#
#   Trainer
#
#
import logging
import os
import mlflow
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as T

from glob import glob
from tqdm import tqdm, trange
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from ....base import BaseTrainer
from .dataset import DCGANDataset
from .hydra_conf import DCGANHydraConf
from .networks import Generator, Discriminator


def build_transform():
    return T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Lambda(lambda img: (img * 2) - 1)  # normalize values between -1 and 1
    ])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@dataclass
class DCGANTrainer(DCGANHydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def __post_init__(self):
        if self.num_workers < 0:
            self.num_workers = mp.cpu_count()
        if self.gpus < 0:
            self.gpus = torch.cuda.device_count()

        assert self.num_epochs > 0

    def prepare_data(self, root):
        logger = logging.getLogger(__name__)

        self._filepaths = (glob(os.path.join(root, "*.png")) + glob(os.path.join(root, "*.jpg")) +
                           glob(os.path.join(root, "*.gif")))
        logger.info(f"Found {len(self._filepaths)} images")

    def train(self):
        device = "cuda" if (torch.cuda.is_available() and self.gpus > 0) else "cpu"

        generator = Generator(latent_size=self.latent_size, num_channels=self.num_channels,
                              depth=self.generator_depth).to(device)
        discriminator = Discriminator(num_channels=self.num_channels, depth=self.discriminator_depth).to(device)

        if self.gpus > 1:
            generator = nn.DataParallel(generator, range(self.gpus))
            discriminator = nn.DataParallel(discriminator, range(self.gpus))

        generator.apply(weights_init)
        discriminator.apply(weights_init)

        generator_optimizer = optim.Adam(generator.parameters(), lr=self.generator_learning_rate,
                                         betas=(self.beta1, 0.999))
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=self.discriminator_learning_rate,
                                             betas=(self.beta1, 0.999))

        transform = build_transform()

        dataset = DCGANDataset(self._filepaths, transform)

        loader = DataLoader(dataset, self.batch_size, num_workers=self.num_workers)

        noise = torch.randn(10, self.latent_size, 1, 1, device=device)

        for epoch in trange(self.num_epochs):
            running_metrics = {}
            for real_imgs in tqdm(loader, leave=False):
                batch_size = len(real_imgs)

                # detach -> we don't need gradients for generator
                fake_imgs = generator(torch.randn(batch_size, self.latent_size, 1, 1, device=device)).detach()
                fake_imgs_preds = discriminator(fake_imgs)
                fake_imgs_loss = F.binary_cross_entropy_with_logits(fake_imgs_preds, torch.zeros(batch_size, 1,
                                                                                                 device=device))

                real_imgs_preds = discriminator(real_imgs.to(device))
                real_imgs_loss = F.binary_cross_entropy_with_logits(real_imgs_preds, torch.ones(batch_size, 1,
                                                                                                device=device))

                discriminator_loss = fake_imgs_loss + real_imgs_loss

                discriminator_optimizer.zero_grad()

                discriminator_loss.backward()

                discriminator_optimizer.step()

                fake_imgs = generator(torch.randn(batch_size, self.latent_size, 1, 1, device=device))
                fake_imgs_preds = discriminator(fake_imgs)
                generator_loss = F.binary_cross_entropy_with_logits(fake_imgs_preds, torch.ones(batch_size, 1,
                                                                                                device=device))

                generator_optimizer.zero_grad()

                generator_loss.backward()

                generator_optimizer.step()

                log_metrics = [
                    ("train/discriminator_loss", discriminator_loss.item()),
                    ("train/discriminator_real_imgs_loss", real_imgs_loss.item()),
                    ("train/discriminator_fake_imgs_loss", fake_imgs_loss.item()),
                    ("train/generator_loss", generator_loss.item())
                ]
                for metric_name, metric_value in log_metrics:
                    running_metrics[metric_name] = running_metrics.get(metric_name, 0) + metric_value

            mean_epoch_metrics = {k: v / len(loader) for k, v in running_metrics.items()}
            mlflow.log_metrics({**mean_epoch_metrics, "epoch": epoch})

            if (epoch == self.num_epochs - 1) or (self.log_images_interval is not None and
                                                  (epoch % self.log_images_interval) == 0):
                with torch.no_grad():
                    fake_imgs = generator(noise)

                fake_imgs = (fake_imgs + 1) / 2  # denormalize

                val_img = to_pil_image(make_grid(fake_imgs, 5)).convert("L" if self.num_channels == 1 else "RGB")
                mlflow.log_image(val_img, f"epoch-{epoch}.png")

        torch.save(generator.state_dict(), f"generator_epoch={epoch}.pt")
        mlflow.log_artifact(f"generator_epoch={epoch}.pt")
