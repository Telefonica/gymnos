#
#
#   Histogram Equalisation
#
#

import warnings

from PIL import ImageOps

from ..utils.image_utils import arr_to_img, img_to_arr
from .data_augmentor import DataAugmentor


class HistogramEqualisation(DataAugmentor):
    """
    The class :class:`HistogramEqualisation` is used to perform histogram
    equalisation on images passed to its :func:`transform` function.

    As there are no further user definable parameters, the class is
    instantiated using only the :attr:`probability` argument.

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :type probability: float
    """

    def __init__(self, probability):
        super().__init__(probability)

    def transform(self, image):
        """
        Performs histogram equalisation on the images passed as an argument
        and returns the equalised images. There are no user definable
        parameters for this method.

        :param image: The image(s) on which to perform the histogram
         equalisation.
        :type image: np.ndarray
        :return: The transformed image
        """
        # If an image is a colour image, the histogram will
        # will be computed on the flattened image, which fires
        # a warning.
        # We may want to apply this instead to each colour channel.
        image = arr_to_img(image)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = ImageOps.equalize(image)

        return img_to_arr(image)
