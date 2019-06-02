#
#
#   Rotate range
#
#

import math
import random
import numpy as np

from .data_augmentor import DataAugmentor

from PIL import Image


class RotateRange(DataAugmentor):
    """
    This class is used to perform rotations on images by arbitrary numbers of
    degrees.

    Images are rotated **in place** and an image of the same size is
    returned by this function. That is to say, that after a rotation
    has been performed, the largest possible area of the same aspect ratio
    of the original image is cropped from the skewed image, and this is
    then resized to match the original image size.

    The method by which this is performed is described as follows:

    .. math::

        E = \\frac{\\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}}\\Big(X-\\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}} Y\\Big)}{1-\\frac{(\\sin{\\theta_{a}})^2}{(\\sin{\\theta_{b}})^2}} # noqa E501

    which describes how :math:`E` is derived, and then follows
    :math:`B = Y - E` and :math:`A = \\frac{\\sin{\\theta_{a}}}{\\sin{\\theta_{b}}} B`.

    The :ref:`rotating` section describes this in detail and has example
    images to demonstrate this.
    """

    def __init__(self, probability, max_left_rotation, max_right_rotation):
        """
        As well as the required :attr:`probability` parameter, the
        :attr:`max_left_rotation` parameter controls the maximum number of
        degrees by which to rotate to the left, while the
        :attr:`max_right_rotation` controls the maximum number of degrees to
        rotate to the right.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param max_left_rotation: The maximum number of degrees to rotate
         the image anti-clockwise.
        :param max_right_rotation: The maximum number of degrees to rotate
         the image clockwise.
        :type probability: Float
        :type max_left_rotation: Integer
        :type max_right_rotation: Integer
        """
        super().__init__(probability)
        self.max_left_rotation = -abs(max_left_rotation)   # Ensure always negative
        self.max_right_rotation = abs(max_right_rotation)  # Ensure always positive

    def transform(self, image):
        """
        Perform the rotation on the passed :attr:`image` and return
        the transformed image. Uses the :attr:`max_left_rotation` and
        :attr:`max_right_rotation` passed into the constructor to control
        the amount of degrees to rotate by. Whether the image is rotated
        clockwise or anti-clockwise is chosen at random.

        :param image: The image(s) to rotate.
        :type image: np.array
        :return: The transformed image
        """
        image = Image.fromarray(image)
        # TODO: Small rotations of 1 or 2 degrees can create black pixels
        random_left = random.randint(self.max_left_rotation, 0)
        random_right = random.randint(0, self.max_right_rotation)

        left_or_right = random.randint(0, 1)

        rotation = 0

        if left_or_right == 0:
            rotation = random_left
        elif left_or_right == 1:
            rotation = random_right

        # Get size before we rotate
        x = image.size[0]
        y = image.size[1]

        # Rotate, while expanding the canvas size
        image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)

        # Get size after rotation, which includes the empty space
        X = image.size[0]
        Y = image.size[1]

        # Get our two angles needed for the calculation of the largest area
        angle_a = abs(rotation)
        angle_b = 90 - angle_a

        # Python deals in radians so get our radians
        angle_a_rad = math.radians(angle_a)
        angle_b_rad = math.radians(angle_b)

        # Find the maximum area of the rectangle that could be cropped
        E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
            (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
        E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
        B = X - E
        A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

        # Crop this area from the rotated image
        # image = image.crop((E, A, X - E, Y - A))
        image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

        # Return the image, re-sized to the size of the image passed originally
        image = image.resize((x, y), resample=Image.BICUBIC)

        return np.array(image)
