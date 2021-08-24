#
#
#   Networks
#
#

import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, depth: int = 64, latent_size: int = 128, num_channels: int = 3):
        super().__init__()

        self._hidden_0 = nn.Sequential(
            nn.ConvTranspose2d(latent_size, depth * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(depth * 8),
            nn.ReLU(inplace=True),
        )

        self._hidden_1 = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth * 4),
            nn.ReLU(inplace=True)
        )

        self._hidden_2 = nn.Sequential(
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth * 2),
            nn.ReLU(inplace=True),
        )

        self._hidden_3 = nn.Sequential(
            nn.ConvTranspose2d(depth * 2, depth, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True),
        )

        self._out = nn.Sequential(
            nn.ConvTranspose2d(depth, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self._hidden_0(x)
        x = self._hidden_1(x)
        x = self._hidden_2(x)
        x = self._hidden_3(x)
        x = self._out(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, depth: int = 64, num_channels: int = 3):
        super().__init__()

        self._hidden_0 = nn.Sequential(
            nn.Conv2d(num_channels, depth, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._hidden_1 = nn.Sequential(
            nn.Conv2d(depth, depth * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._hidden_2 = nn.Sequential(
            nn.Conv2d(depth * 2, depth * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._hidden_3 = nn.Sequential(
            nn.Conv2d(depth * 4, depth * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depth * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self._out = nn.Conv2d(depth * 8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self._hidden_0(x)
        x = self._hidden_1(x)
        x = self._hidden_2(x)
        x = self._hidden_3(x)
        x = self._out(x)
        x = x.view(-1, 1)
        return x
