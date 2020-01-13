#
#
#   Zoom
#
#

import math
import random

from ..utils.iterator_utils import apply
from .data_augmentor import DataAugmentor
from ..utils.lazy_imports import lazy_imports as lazy
from ..preprocessors.utils.image_ops import arr_to_img, img_to_arr


class Zoom(DataAugmentor):
    """
    This class is used to enlarge images (to zoom) but to return a cropped
    region of the zoomed image of the same size as the original image.

    The amount of zoom applied is randomised, from between
    :attr:`min_factor` and :attr:`max_factor`. Set these both to the same
    value to always zoom by a constant factor.

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :param min_factor: The minimum amount of zoom to apply. Set both the
     :attr:`min_factor` and :attr:`min_factor` to the same values to zoom
     by a constant factor.
    :param max_factor: The maximum amount of zoom to apply. Set both the
     :attr:`min_factor` and :attr:`min_factor` to the same values to zoom
     by a constant factor.
    :type probability: float
    :type min_factor: float
    :type max_factor: float
    """

    def __init__(self, probability, min_factor, max_factor):
        super().__init__(probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def transform(self, images):
        """
        Zooms/scales the passed image(s) and returns the new image.

        :param image: The image(s) to be zoomed.
        :type image: np.array
        :return: The transformed image
        """
        def operation(image):
            image = arr_to_img(image)
            factor = round(random.uniform(self.min_factor, self.max_factor), 2)

            w, h = image.size

            image_zoomed = image.resize((int(round(image.size[0] * factor)),
                                         int(round(image.size[1] * factor))),
                                        resample=lazy.PIL.Image.BICUBIC)
            w_zoomed, h_zoomed = image_zoomed.size

            image = image_zoomed.crop((math.floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                       math.floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                       math.floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                       math.floor((float(h_zoomed) / 2) + (float(h) / 2))))

            return img_to_arr(image)

        return apply(images, operation)
