#
#
#   GrayScale
#
#

from ..utils.iterator_utils import apply
from ..utils.lazy_imports import lazy_imports as lazy
from ..preprocessors.utils.image_ops import arr_to_img, img_to_arr
from .data_augmentor import DataAugmentor


class Grayscale(DataAugmentor):
    """
    This class is used to convert images into grayscale. That is, it converts
    images into having only shades of grey (pixel value intensities)
    varying from 0 to 255 which represent black and white respectively.

    As there are no further user definable parameters, the class is
    instantiated using only the :attr:`probability` argument.

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :type probability: Float
    """

    def transform(self, images):
        """
        Converts the passed image to grayscale and returns the transformed
        image. There are no user definable parameters for this method.

        :param image: The image to convert to grayscale.
        :type image: np.ndarray
        :return: The transformed image
        """
        PIL = __import__("{}.ImageOps".format(lazy.PIL.__name__))

        def operation(image):
            image = arr_to_img(image)
            new_image = PIL.ImageOps.grayscale(image)
            new_image = new_image.convert(image.mode)
            return img_to_arr(new_image)

        return apply(images, operation)
