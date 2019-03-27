#
#
#   Grayscale preprocessor
#
#

from ..utils.image_utils import imgray
from ..utils.iterator_utils import apply
from .preprocessor import Preprocessor


class Grayscale(Preprocessor):

    def __transform_sample(self, x):
        return imgray(x)

    def transform(self, X):
        return apply(X, self.__transform_sample)
