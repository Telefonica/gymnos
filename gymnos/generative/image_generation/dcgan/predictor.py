#
#
#   Predictor
#
#

import os
import glob
import torch
import warnings
import numpy as np

from omegaconf import DictConfig
from torchvision.transforms.functional import to_pil_image

from .networks import Generator
from ....utils.py_utils import lmap
from ....base import BasePredictor, MLFlowRun


def img_tensor_to_array(img_tensor):
    return np.array(to_pil_image(img_tensor))


class DCGANPredictor(BasePredictor):
    """
    Parameters
    ------------
    device
        Device to run predictions, e.g ``cuda``, ``cpu`` or ``cuda:0``. If ``auto``, ``cuda`` will be used if
        CUDA is available otherwise ``cpu`` will be used.
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self._generator = None

    def load(self, config: DictConfig, run: MLFlowRun, artifacts_dir: str):
        checkpoints = glob.glob(os.path.join(artifacts_dir, "*.pt"))
        if len(checkpoints) == 0:
            raise ValueError("No checkpoint found")
        if len(checkpoints) > 1:
            warnings.warn("More than one checkpoint found. Selecting the first one")

        self._generator = Generator(latent_size=config.trainer.latent_size, num_channels=config.trainer.num_channels,
                                    depth=config.trainer.generator_depth)
        state_dict = torch.load(checkpoints[0], map_location=self.device)
        # I'm not sure why state_dict is prefixed with module but we need to remove it
        state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
        self._generator.load_state_dict(state_dict)
        self._generator.eval().to(self.device)

    @torch.no_grad()
    def predict(self, latent_vector: np.ndarray):
        """
        Generate fake images

        Parameters
        ----------
        latent_vector: np.ndarray
            Latent vector, must be a NumPy array with the following shape (B, L, 1, 1) where:

                - B: batch size (number of images to generate)
                - L: latent size.

        Returns
        -------
        np.ndarray
            NumPy Array of images with the following shape (B, H, W) if the image has only one channel and
            (B, H, W, C) if the image has more than one channel
        """
        fake_imgs = self._generator(torch.from_numpy(latent_vector).float().to(self.device))
        fake_imgs = (fake_imgs + 1) / 2  # denormalize
        np_imgs = lmap(img_tensor_to_array, fake_imgs)
        return np.array(np_imgs)
