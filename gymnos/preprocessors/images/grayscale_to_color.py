#
#
#   Grayscale to Color
#
#

import numpy as np

from ...utils.iterator_utils import apply
from ..preprocessor import Preprocessor


class GrayscaleToColor(Preprocessor):
    """
    Convert grayscale to color (1D image to 3D image).
    """

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def __transform_sample(self, x):
        return np.squeeze(np.stack([x] * 3, -1))

    def transform(self, X):
        return apply(X, self.__transform_sample)