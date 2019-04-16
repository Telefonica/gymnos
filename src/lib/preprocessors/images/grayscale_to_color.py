#
#
#   Grayscale to Color
#
#

import cv2 as cv

from ...utils.iterator_utils import apply
from ..preprocessor import Preprocessor


class GrayscaleToColor(Preprocessor):

    def __transform_sample(self, x):
        return cv.cvtColor(x, cv.COLOR_GRAY2RGB)

    def transform(self, X):
        return apply(X, self.__transform_sample)
