#
#
#   Test grayscale
#
#

import numpy as np

from gymnos.preprocessors.images.grayscale import Grayscale


def test_transform():
    image = np.random.randint(0, 255, [120, 120, 3], dtype=np.uint8)
    image = image[np.newaxis]

    assert image.shape == (1, 120, 120, 3)

    grayscale = Grayscale()
    new_images = grayscale.transform(image)
    assert new_images.shape == (1, 120, 120, 1)
