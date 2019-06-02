#
#
#   Rotate standard
#
#

import random
import numpy as np

from PIL import Image

from .data_augmentor import DataAugmentor


class RotateStandard(DataAugmentor):
    """
    Class to perform rotations without automatically cropping the images,
    as opposed to the :class:`RotateRange` class.

    .. seealso:: For arbitrary rotations with automatic cropping, see
     the :class:`RotateRange` class.
    .. seealso:: For 90 degree rotations, see the :class:`Rotate` class.
    """

    def __init__(self, probability, max_left_rotation, max_right_rotation, expand=False):
        """
        Documentation to appear.
        """
        super().__init__(probability)
        self.max_left_rotation = -abs(max_left_rotation)   # Ensure always negative
        self.max_right_rotation = abs(max_right_rotation)  # Ensure always positive
        self.expand = expand

    def transform(self, image):
        """
        Documentation to appear.

        :type image: np.array
        :return: The transformed image
        """
        image = Image.fromarray(image)
        random_left = random.randint(self.max_left_rotation, 0)
        random_right = random.randint(0, self.max_right_rotation)

        left_or_right = random.randint(0, 1)

        rotation = 0

        if left_or_right == 0:
            rotation = random_left
        elif left_or_right == 1:
            rotation = random_right

        image = image.rotate(rotation, expand=self.expand, resample=Image.BICUBIC)

        return np.array(image)
