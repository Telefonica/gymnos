#
#
#   Invert
#
#

from .data_augmentor import DataAugmentor

from PIL import ImageOps
from ..utils.image_utils import arr_to_img, img_to_arr


class Invert(DataAugmentor):
    """
    This class is used to negate images. That is to reverse the pixel values
    for any image processed by it.
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
        Negates the image passed as an argument. There are no user definable
        parameters for this method.

        :param image: The image(s) to negate.
        :type image: np.array
        :return: The transformed image
        """
        image = arr_to_img(image)
        image = ImageOps.invert(image)
        return img_to_arr(image)
