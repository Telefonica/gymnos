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

from ....base import BasePredictor, MLFlowRun

from torchvision.transforms.functional import to_pil_image
from .networks import Generator
from ....utils.py_utils import lmap
from ....base import BasePredictor, MLFlowRun

def img_tensor_to_array(img_tensor):
    return np.array(to_pil_image(img_tensor))
    
class WganGpPredictor(BasePredictor):

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

        self._gen = Generator(channels_noise=config.trainer.z_dim, channels_img=config.trainer.channels_img,
                              features_g=config.trainer.features_g)
        state_dict = torch.load(checkpoints[0], map_location=self.device)
        state_dict = {k.partition("module.")[2]: state_dict[k] for k in state_dict.keys()}
        self._gen.load_state_dict(state_dict)
        self._gen.eval().to(self.device)

    @torch.no_grad()
    def predict(self, latent_vector: np.ndarray):
        fake_imgs = self._gen(torch.from_numpy(latent_vector).float().to(self.device))
        fake_imgs = (fake_imgs + 1) / 2  # denormalize
        np_imgs = lmap(img_tensor_to_array, fake_imgs)

        return np.array(np_imgs)
