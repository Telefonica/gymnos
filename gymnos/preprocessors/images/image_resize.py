#
#
#   Image resize preprocessor
#
#

from ..preprocessor import Preprocessor
from ..utils.image_ops import imresize
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

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def _transform_sample(self, x):
        resized = imresize(x, (self.width, self.height))
        return resized

    def transform(self, X):
        return apply(X, self._transform_sample)

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
