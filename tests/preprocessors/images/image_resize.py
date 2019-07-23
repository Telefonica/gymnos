#
#
#   Image resize
#
#

import numpy as np

from gymnos.preprocessors import ImageResize


def test_transform(random_rgb_image, random_gray_image):
    resizer = ImageResize(150, 150)

    new_image = resizer.transform(random_rgb_image[np.newaxis])

    assert new_image.shape[1:] == (150, 150, 3)

    new_image = resizer.transform(random_gray_image[np.newaxis])

    assert new_image.shape[1:] == (150, 150, 1)
