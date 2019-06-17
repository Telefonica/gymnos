#
#
#   Random brightness
#
#

import numpy as np

from PIL import ImageEnhance

from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr


class RandomBrightness(DataAugmentor):
    """
    This class is used to random change image brightness.
    """

    def __init__(self, probability, min_factor, max_factor):
        """
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

        :type probability: Float
        :type max_factor: Float
        :type max_factor: Float
        """
        super().__init__(probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def transform(self, image):
        """
        Random change the passed image brightness.

        :param image: The image to convert into monochrome.
        :type image: np.array
        :return: The transformed image
        """
        image = arr_to_img(image)
        factor = np.random.uniform(self.min_factor, self.max_factor)

        image_enhancer_brightness = ImageEnhance.Brightness(image)
        image = image_enhancer_brightness.enhance(factor)
        return img_to_arr(image)
