#
#
#   GreyScale
#
#

import numpy as np

from PIL import ImageOps, Image

from .data_augmentor import DataAugmentor


class Greyscale(DataAugmentor):
    """
    This class is used to convert images into greyscale. That is, it converts
    images into having only shades of grey (pixel value intensities)
    varying from 0 to 255 which represent black and white respectively.

    .. seealso:: The :class:`BlackAndWhite` class.
    """

    def __init__(self, probability):
        """
        As there are no further user definable parameters, the class is
        instantiated using only the :attr:`probability` argument.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        super().__init__(probability)

    def transform(self, image):
        """
        Converts the passed image to greyscale and returns the transformed
        image. There are no user definable parameters for this method.

        :param image: The image to convert to greyscale.
        :type image: np.array
        :return: The transformed image
        """
        image = Image.fromarray(image)
        image = ImageOps.grayscale(image)
        return np.array(image)
