#
#
#   Flip
#
#

import random

from ..utils.iterator_utils import apply
from ..utils.lazy_imports import lazy_imports as lazy
from ..preprocessors.utils.image_ops import arr_to_img, img_to_arr
from .data_augmentor import DataAugmentor


class Flip(DataAugmentor):
    """
    This class is used to mirror images through the x or y axes.

    The class allows an image to be mirrored along either
    its x axis or its y axis, or randomly.
    The direction of the flip, or whether it should be randomised, is
    controlled using the :attr:`top_bottom_left_right` parameter.

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :type probability: float
    :type top_bottom_left_right: str
    :param top_bottom_left_right: Controls the direction the image should
     be mirrored. Must be one of ``LEFT_RIGHT``, ``TOP_BOTTOM``, or
     ``RANDOM``.

     - ``LEFT_RIGHT`` defines that the image is mirrored along its x axis.
     - ``TOP_BOTTOM`` defines that the image is mirrored along its y axis.
     - ``RANDOM`` defines that the image is mirrored randomly along
       either the x or y axis.
    """

    def __init__(self, probability, top_bottom_left_right):
        super().__init__(probability)
        self.top_bottom_left_right = top_bottom_left_right

    def transform(self, images):
        """
        Mirror the image according to the `attr`:top_bottom_left_right`
        argument passed to the constructor and return the mirrored image.

        :param image: The image to mirror.
        :type image: np.ndarray
        :return: The transformed image
        """
        def operation(image):
            image = arr_to_img(image)
            random_axis = random.randint(0, 1)

            if self.top_bottom_left_right == "LEFT_RIGHT":
                image = image.transpose(lazy.PIL.Image.FLIP_LEFT_RIGHT)
            elif self.top_bottom_left_right == "TOP_BOTTOM":
                image = image.transpose(lazy.PIL.Image.FLIP_TOP_BOTTOM)
            elif self.top_bottom_left_right == "RANDOM":
                if random_axis == 0:
                    image = image.transpose(lazy.PIL.Image.FLIP_LEFT_RIGHT)
                elif random_axis == 1:
                    image = image.transpose(lazy.PIL.Image.FLIP_TOP_BOTTOM)

            return img_to_arr(image)

        return apply(images, operation)
