#
#
#   Black and white
#
#

import numpy as np

from PIL import Image, ImageOps

from .data_augmentor import DataAugmentor


class BlackAndWhite(DataAugmentor):
    """
    This class is used to convert images into black and white. In other words,
    into using a 1-bit, monochrome binary colour palette. This is not to be
    confused with greyscale, where an 8-bit greyscale pixel intensity range
    is used.

    .. seealso:: The :class:`Greyscale` class.
    """

    def __init__(self, probability, threshold):
        """
        As well as the required :attr:`probability` parameter, a
        :attr:`threshold` can also be defined to define the cutoff point where
        a pixel is converted to black or white. The :attr:`threshold` defaults
        to 128 at the user-facing
        :func:`~Augmentor.Pipeline.Pipeline.black_and_white` function.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param threshold: A value between 0 and 255 that defines the cut off
         point where an individual pixel is converted into black or white.
        :type probability: Float
        :type threshold: Integer
        """
        super().__init__(probability)
        self.threshold = threshold

    def transform(self, image):
        """
        Convert the image passed as an argument to black and white, 1-bit
        monochrome. Uses the :attr:`threshold` passed to the constructor
        to control the cut-off point where a pixel is converted to black or
        white.

        :param image: The image to convert into monochrome.
        :type image: np.array.
        :return: The transformed image
        """

        # An alternative would be to use
        # PIL.ImageOps.posterize(image=image, bits=1)
        # but this might be faster.
        image = Image.fromarray(image)
        image = ImageOps.grayscale(image)
        image = image.point(lambda x: 0 if x < self.threshold else 255, '1')
        return np.array(image)
