#
#
#   Zoom ground truth
#
#

import math
import random

from PIL import Image

from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr


class ZoomGroundTruth(DataAugmentor):
    """
    This class is used to enlarge images (to zoom) but to return a cropped
    region of the zoomed image of the same size as the original image.
    """

    def __init__(self, probability, min_factor, max_factor):
        """
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
        :type probability: Float
        :type min_factor: Float
        :type max_factor: Float
        """
        super().__init__(probability)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def transform(self, image):
        """
        Zooms/scales the passed images and returns the new images.

        :param image: An arbitrarily long list of image(s) to be zoomed.
        :type image: np.array
        :return: The zoomed in image.
        """
        image = arr_to_img(image)
        factor = round(random.uniform(self.min_factor, self.max_factor), 2)

        w, h = image.size

        # TODO: Join these two functions together so that we don't have this image_zoom variable lying around.
        image_zoomed = image.resize((int(round(image.size[0] * factor)), int(round(image.size[1] * factor))),
                                    resample=Image.BICUBIC)
        w_zoomed, h_zoomed = image_zoomed.size

        image = image_zoomed.crop((math.floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                   math.floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                   math.floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                   math.floor((float(h_zoomed) / 2) + (float(h) / 2))))

        return img_to_arr(image)
