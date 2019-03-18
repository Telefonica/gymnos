#
#
#   Image resize preprocessor
#
#

import numpy as np

from .preprocessor import Preprocessor
from ..utils.image_utils import imresize
from ..utils.iterator_utils import apply


class ImageResize(Preprocessor):

    def __init__(self, width, height):
        self.width = width
        self.height = height


    def transform_sample(self, x):
        resized = imresize(x, (self.width, self.height))

        if resized.ndim < 3:
            resized = resized[..., np.newaxis]

        return resized


    def transform(self, X):
        return apply(X, self.transform_sample)
