#
#
#   Grayscale preprocessor
#
#

from ...utils.image_utils import imgray
from ...utils.iterator_utils import apply
from ..preprocessor import Preprocessor


class Grayscale(Preprocessor):
    """
    Convert color images to grayscale.
    """

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def __transform_sample(self, x):
        return imgray(x)

    def transform(self, X):
        return apply(X, self.__transform_sample)
