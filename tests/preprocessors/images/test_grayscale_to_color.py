#
#
#   Test grayscale to color
#
#

import numpy as np

from gymnos.preprocessors import GrayscaleToColor


def test_transform(random_gray_image):
    grayscale_to_color = GrayscaleToColor()

    assert random_gray_image.shape[-1] == 1

    new_image = grayscale_to_color.transform(random_gray_image[np.newaxis])

    assert new_image.shape[-1] == 3
