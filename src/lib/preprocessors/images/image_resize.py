#
#
#   Image resize preprocessor
#
#

import numpy as np

from ..preprocessor import Preprocessor
from ...utils.image_utils import imresize
from ...utils.iterator_utils import apply


class ImageResize(Preprocessor):
    """
    Resize image

    Parameters
    ----------

    width: int or float, optional
        Width of the new image
    height: int or float, optional
        Height of the new image
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height


    def __transform_sample(self, x):
        resized = imresize(x, (self.width, self.height))

        if resized.ndim < 3:
            resized = resized[..., np.newaxis]

        return resized


    def transform(self, X):
        return apply(X, self.__transform_sample)
