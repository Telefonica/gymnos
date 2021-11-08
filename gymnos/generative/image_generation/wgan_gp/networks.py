#
#
#   Networks
#
#

import torch
import torch.nn as nn

# -----------------Dicriminator or Critic-----------------


class Critic(nn.Module):
    def __init__(self, channels_img: int = 3, features_c: int = 16):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(

            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_c, features_c * 2, 4, 2, 1),
            self._block(features_c * 2, features_c * 4, 4, 2, 1),
            self._block(features_c * 4, features_c * 8, 4, 2, 1),

            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_c * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.crit(x)

# -----------------Generator-----------------


class Generator(nn.Module):
    def __init__(self, channels_noise: int = 100, channels_img: int = 3, features_g: int = 16):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
