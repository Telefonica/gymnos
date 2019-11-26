#
#
#   Invert
#
#

from PIL import ImageOps

from ..utils.iterator_utils import apply
from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr


class Invert(DataAugmentor):
    """
    This class is used to negate images. That is to reverse the pixel values
    for any image processed by it.

    As there are no further user definable parameters, the class is
    instantiated using only the :attr:`probability` argument.

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :type probability: float
    """

    def __init__(self, probability):
        super().__init__(probability)

    def transform(self, images):
        """
        Negates the image passed as an argument. There are no user definable
        parameters for this method.

        :param image: The image(s) to negate.
        :type image: np.array
        :return: The transformed image
        """
        def operation(image):
            image = arr_to_img(image)
            image = ImageOps.invert(image)
            return img_to_arr(image)

        return apply(images, operation)
