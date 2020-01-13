#
#
#   Random brightness
#
#

import numpy as np

from ..utils.iterator_utils import apply
from .data_augmentor import DataAugmentor
from ..utils.lazy_imports import lazy_imports as lazy
from ..preprocessors.utils.image_ops import arr_to_img, img_to_arr


class RandomBrightness(DataAugmentor):
    """
    This class is used to random change image brightness.

    required :attr:`probability` parameter

    :func:`~Augmentor.Pipeline.Pipeline.random_brightness` function.

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :param min_factor: The value between 0.0 and max_factor
     that define the minimum adjustment of image brightness.
     The value  0.0 gives a black image,The value 1.0 gives the original image, value bigger than 1.0 gives more bright image. # noqa E501
    :param max_factor: A value should be bigger than min_factor.
     that define the maximum adjustment of image brightness.
     The value  0.0 gives a black image, value 1.0 gives the original image, value bigger than 1.0 gives more bright image. # noqa E501

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
        Random change the passed image brightness.

        :param image: The image to convert into monochrome.
        :type image: np.array
        :return: The transformed image
        """
        PIL = __import__("{}.ImageEnhance".format(lazy.PIL.__name__))

        def operation(image):
            image = arr_to_img(image)
            factor = np.random.uniform(self.min_factor, self.max_factor)

            image_enhancer_brightness = PIL.ImageEnhance.Brightness(image)
            image = image_enhancer_brightness.enhance(factor)
            return img_to_arr(image)

        return apply(images, operation)
