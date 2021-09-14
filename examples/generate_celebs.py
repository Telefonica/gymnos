#
#
#   Generate celebs
#
#

from gymnos.utils import assert_dependencies

assert_dependencies([
    "Pillow",
    "matplotlib",
    "gymnos[generative.image_generation.dcgan]"
])

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from gymnos.utils import lmap
from gymnos.generative.image_generation.dcgan import DCGANPredictor


def imgrid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


predictor = DCGANPredictor.from_pretrained("ruben/models/celeba-dcgan")

latent_vector = np.random.randn(9, 128, 1, 1)

fake_imgs = predictor.predict(latent_vector)

grid_fake_imgs = imgrid(lmap(Image.fromarray, fake_imgs), 3, 3)

plt.imshow(grid_fake_imgs)
plt.show()
