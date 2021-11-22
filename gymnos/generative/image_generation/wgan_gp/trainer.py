#
#
#   Trainer
#
#
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#import torchvision.datasets as datasets
import torchvision.transforms as transforms


import os
import mlflow
import logging
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
from .dataset import WGANGPDataset
from .utils import gradient_penalty
from .hydra_conf import WganGpHydraConf
from .networks import Generator, Critic


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def build_transform():
    return T.Compose([
        transforms.Resize(WganGpHydraConf.image_size),
        transforms.RandomCrop(WganGpHydraConf.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(WganGpHydraConf.channels_img)], [0.5 for _ in range(WganGpHydraConf.channels_img)]),
    ])


@dataclass
class WganGpTrainer(WganGpHydraConf, BaseTrainer):
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

        gen = Generator(channels_noise=self.z_dim, channels_img=self.channels_img,
                        features_g=self.features_g).to(device)
        critic = Critic(channels_img=self.channels_img, features_c=self.features_c).to(device)

        initialize_weights(gen)
        initialize_weights(critic)

        opt_gen = optim.Adam(gen.parameters(), lr=self.learning_rate, betas=(0.0, 0.9))
        opt_critic = optim.Adam(critic.parameters(), lr=self.learning_rate, betas=(0.0, 0.9))

        transform = build_transform()

        dataset = WGANGPDataset(self._filepaths, transform)
        #dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        noise = torch.randn(10, self.z_dim, 1, 1, device=device)

        for epoch in range(self.num_epochs):
            running_metrics = {}

            for real in tqdm(loader, leave=False):
                real = real.to(device)
                cur_batch_size = real.shape[0]

                for _ in range(self.critic_iterations):
                    noise = torch.randn(cur_batch_size, self.z_dim, 1, 1).to(device)
                    fake = gen(noise)
                    critic_real = critic(real).reshape(-1)
                    critic_fake = critic(fake).reshape(-1)
                    gp = gradient_penalty(critic, real, fake, device=device)
                    loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gp * gp)
                    critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    opt_critic.step()

                gen_fake = critic(fake).reshape(-1)
                loss_gen = -torch.mean(gen_fake)
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                log_metrics = [
                    ("train/discriminator_loss", loss_critic.item()),
                    ("train/generator_loss", loss_gen.item())
                ]
                for metric_name, metric_value in log_metrics:
                    running_metrics[metric_name] = running_metrics.get(metric_name, 0) + metric_value

            mean_epoch_metrics = {k: v / len(loader) for k, v in running_metrics.items()}
            mlflow.log_metrics({**mean_epoch_metrics, "epoch": epoch})

            if (epoch == self.num_epochs - 1) or ((epoch % 10) == 0):
                with torch.no_grad():
                    fake_imgs = gen(noise)

                fake_imgs = (fake_imgs + 1) / 2  # denormalize

                val_img = to_pil_image(make_grid(fake_imgs, 5)).convert("L" if self.channels_img == 1 else "RGB")
                mlflow.log_image(val_img, f"epoch-{epoch}.png")

        torch.save(generator.state_dict(), f"generator_epoch={epoch}.pt")
        mlflow.log_artifact(f"generator_epoch={epoch}.pt")

    def test(self):
        pass   # OPTIONAL: test code
