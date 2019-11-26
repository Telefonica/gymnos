#
#
#   Rotate
#
#

import random

from ..utils.iterator_utils import apply
from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr


class Rotate(DataAugmentor):
    """
    This class is used to perform rotations on images in multiples of 90
    degrees. Arbitrary rotations are handled by the :class:`RotateRange`
    class.

    As well as the required :attr:`probability` parameter, the
    :attr:`rotation` parameter controls the rotation to perform,
    which must be one of ``90``, ``180``, ``270`` or ``-1`` (see below).

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :param rotation: Controls the rotation to perform. Must be one of
     ``90``, ``180``, ``270`` or ``-1``.

     - ``90`` rotate the image by 90 degrees.
     - ``180`` rotate the image by 180 degrees.
     - ``270`` rotate the image by 270 degrees.
     - ``-1`` rotate the image randomly by either 90, 180, or 270 degrees.
    """

    def __init__(self, probability, rotation):
        super().__init__(probability)
        self.rotation = rotation

    def __str__(self):
        return "Rotate " + str(self.rotation)

    def transform(self, images):
        """
        Rotate an image by either 90, 180, or 270 degrees, or randomly from
        any of these.

        :param image: The image(s) to rotate.
        :type image: np.array
        :return: The transformed image
        """
        def operation(image):
            image = arr_to_img(image)
            random_factor = random.randint(1, 3)

            if self.rotation == -1:
                image = image.rotate(90 * random_factor, expand=True)
            else:
                image = image.rotate(self.rotation, expand=True)

            return img_to_arr(image)

        return apply(images, operation)
