#
#
#   Resize
#
#

import numpy as np

from PIL import Image

from .data_augmentor import DataAugmentor


class Resize(DataAugmentor):
    """
    This class is used to resize images by absolute values passed as parameters.
    """

    def __init__(self, probability, width, height, resample_filter):
        """
        Accepts the required probability parameter as well as parameters
        to control the size of the transformed image.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param width: The width in pixels to resize the image to.
        :param height: The height in pixels to resize the image to.
        :param resample_filter: The resample filter to use. Must be one of
         the standard PIL types, i.e. ``NEAREST``, ``BICUBIC``, ``ANTIALIAS``,
         or ``BILINEAR``.
        :type probability: Float
        :type width: Integer
        :type height: Integer
        :type resample_filter: String
        """
        super().__init__(probability)
        self.width = width
        self.height = height
        self.resample_filter = resample_filter

    def transform(self, image):
        """
        Resize the passed image and returns the resized image. Uses the
        parameters passed to the constructor to resize the passed image.

        :param image: The image to resize.
        :type image: np.array
        :return: The transformed image
        """
        image = Image.fromarray(image)
        # TODO: Automatically change this to ANTIALIAS or BICUBIC depending on the size of the file
        image = image.resize((self.width, self.height), eval("Image.%s" % self.resample_filter))
        return np.array(image)
