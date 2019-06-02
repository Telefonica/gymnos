#
#
#   Crop Random
#
#

import math
import random
import numpy as np

from PIL import Image

from .data_augmentor import DataAugmentor


class CropRandom(DataAugmentor):
    """
    .. warning:: This :class:`CropRandom` class is currently not used by any
     of the user-facing functions in the :class:`~Augmentor.Pipeline.Pipeline`
     class.
    """

    def __init__(self, probability, percentage_area):
        """
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param percentage_area: The percentage area of the original image
         to crop. A value of 0.5 would crop an area that is 50% of the area
         of the original image's size.
        """
        super().__init__(probability)
        self.percentage_area = percentage_area

    def transform(self, image):
        """
        Randomly crop the passed image, returning the crop as a new image.

        :param image: The image to crop.
        :type image: np.array
        :return: The transformed image
        """
        image = Image.fromarray(image)
        w, h = image.size

        w_new = int(math.floor(w * self.percentage_area))
        h_new = int(math.floor(h * self.percentage_area))

        random_left_shift = random.randint(0, int((w - w_new)))  # Note: randint() is from uniform distribution.
        random_down_shift = random.randint(0, int((h - h_new)))

        image = image.crop((random_left_shift, random_down_shift, w_new + random_left_shift,
                            h_new + random_down_shift))

        return np.array(image)
