#
#
#   Random color
#
#

import numpy as np

from ..utils.iterator_utils import apply
from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr


from PIL import ImageEnhance


class RandomColor(DataAugmentor):
    """
    This class is used to random change saturation of an image.

    required :attr:`probability` parameter

    :func:`~Augmentor.Pipeline.Pipeline.random_color` function.

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :param min_factor: The value between 0.0 and max_factor
     that define the minimum adjustment of image saturation.
     The value 0.0 gives a black and white image, value 1.0 gives the original image.
    :param max_factor: A value should be bigger than min_factor.
     that define the maximum adjustment of image saturation.
     The value 0.0 gives a black and white image, value 1.0 gives the original image.

    :type probability: float
    :type max_factor: float
    :type max_factor: float
    """

    def __init__(self, probability, min_factor, max_factor):
        super().__init__(probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def transform(self, images):
        """
        Random change the passed image saturation.

        :param image: The image to convert into monochrome.
        :type image: np.array
        :return: The transformed image
        """
        def operation(image):
            image = arr_to_img(image)
            factor = np.random.uniform(self.min_factor, self.max_factor)

            image_enhancer_color = ImageEnhance.Color(image)
            image = image_enhancer_color.enhance(factor)
            return img_to_arr(image)

        return apply(images, operation)
