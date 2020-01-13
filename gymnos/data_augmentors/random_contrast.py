#
#
#   Random contrast
#
#

import numpy as np

from ..utils.iterator_utils import apply
from .data_augmentor import DataAugmentor
from ..utils.lazy_imports import lazy_imports as lazy
from ..preprocessors.utils.image_ops import arr_to_img, img_to_arr


class RandomContrast(DataAugmentor):
    """
    This class is used to random change contrast of an image.

    required :attr:`probability` parameter

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :param min_factor: The value between 0.0 and max_factor
     that define the minimum adjustment of image contrast.
     The value  0.0 gives s solid grey image, value 1.0 gives the original image.
    :param max_factor: A value should be bigger than min_factor.
     that define the maximum adjustment of image contrast.
     The value  0.0 gives s solid grey image, value 1.0 gives the original image.

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
        Random change the passed image contrast.

        :param image: The image to convert into monochrome.
        :type image: np.array
        :return: The transformed image
        """
        PIL = __import__("{}.ImageEnhance".format(lazy.PIL.__name__))

        def operation(image):
            image = arr_to_img(image)
            factor = np.random.uniform(self.min_factor, self.max_factor)

            image_enhancer_contrast = PIL.ImageEnhance.Contrast(image)
            image = image_enhancer_contrast.enhance(factor)
            return img_to_arr(image)

        return apply(images, operation)
