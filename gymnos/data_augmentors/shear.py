#
#
#   Shear
#
#

import math
import random

from PIL import Image

from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr


class Shear(DataAugmentor):
    """
    This class is used to shear images, that is to tilt them in a certain
    direction. Tilting can occur along either the x- or y-axis and in both
    directions (i.e. left or right along the x-axis, up or down along the
    y-axis).

    Images are sheared **in place** and an image of the same size as the input
    image is returned by this class. That is to say, that after a shear
    has been performed, the largest possible area of the same aspect ratio
    of the original image is cropped from the sheared image, and this is
    then resized to match the original image size.
    """

    def __init__(self, probability, max_shear_left, max_shear_right):
        """
        The shearing is randomised in magnitude, from 0 to the
        :attr:`max_shear_left` or 0 to :attr:`max_shear_right` where the
        direction is randomised. The shear axis is also randomised
        i.e. if it shears up/down along the y-axis or
        left/right along the x-axis.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param max_shear_left: The maximum shear to the left.
        :param max_shear_right: The maximum shear to the right.
        :type probability: Float
        :type max_shear_left: Integer
        :type max_shear_right: Integer
        """
        super().__init__(probability)
        self.max_shear_left = max_shear_left
        self.max_shear_right = max_shear_right

    def transform(self, image):
        """
        Shears the passed image according to the parameters defined during
        instantiation, and returns the sheared image.

        :param image: The image to shear.
        :type image: np.array
        :return: The transformed image
        """
        ######################################################################
        # Old version which uses SciKit Image
        ######################################################################
        # We will use scikit-image for this so first convert to a matrix
        # using NumPy
        # amount_to_shear = round(random.uniform(self.max_shear_left, self.max_shear_right), 2)
        # image_array = img_to_arr(image)
        # And here we are using SciKit Image's `transform` class.
        # shear_transformer = transform.AffineTransform(shear=amount_to_shear)
        # image_sheared = transform.warp(image_array, shear_transformer)
        #
        # Because of warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     return Image.fromarray(img_as_ubyte(image_sheared))
        ######################################################################
        image = arr_to_img(image)
        width, height = image.size

        # For testing.
        # max_shear_left = 20
        # max_shear_right = 20

        angle_to_shear = int(random.uniform((abs(self.max_shear_left) * -1) - 1, self.max_shear_right + 1))
        if angle_to_shear != -1:
            angle_to_shear += 1

        # Alternative method
        # Calculate our offset when cropping
        # We know one angle, phi (angle_to_shear)
        # We known theta = 180-90-phi
        # We know one side, opposite (height of image)
        # Adjacent is therefore:
        # tan(theta) = opposite / adjacent
        # A = opposite / tan(theta)
        # theta = math.radians(180-90-angle_to_shear)
        # A = height / math.tan(theta)

        # Transformation matrices can be found here:
        # https://en.wikipedia.org/wiki/Transformation_matrix
        # The PIL affine transform expects the first two rows of
        # any of the affine transformation matrices, seen here:
        # https://en.wikipedia.org/wiki/Transformation_matrix#/media/File:2D_affine_transformation_matrix.svg

        directions = ["x", "y"]
        direction = random.choice(directions)

        # We use the angle phi in radians later
        phi = math.tan(math.radians(angle_to_shear))

        if direction == "x":
            # Here we need the unknown b, where a is
            # the height of the image and phi is the
            # angle we want to shear (our knowns):
            # b = tan(phi) * a
            shift_in_pixels = phi * height

            if shift_in_pixels > 0:
                shift_in_pixels = math.ceil(shift_in_pixels)
            else:
                shift_in_pixels = math.floor(shift_in_pixels)

            # For negative tilts, we reverse phi and set offset to 0
            # Also matrix offset differs from pixel shift for neg
            # but not for pos so we will copy this value in case
            # we need to change it
            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            # Note: PIL expects the inverse scale, so 1/scale_factor for example.
            transform_matrix = (1, phi, -matrix_offset,
                                0, 1, 0)

            image = image.transform((int(round(width + shift_in_pixels)), height),
                                    Image.AFFINE,
                                    transform_matrix,
                                    Image.BICUBIC)

            image = image.crop((abs(shift_in_pixels), 0, width, height))

            image = image.resize((width, height), resample=Image.BICUBIC)

        elif direction == "y":
            shift_in_pixels = phi * width

            matrix_offset = shift_in_pixels
            if angle_to_shear <= 0:
                shift_in_pixels = abs(shift_in_pixels)
                matrix_offset = 0
                phi = abs(phi) * -1

            transform_matrix = (1, 0, 0,
                                phi, 1, -matrix_offset)

            image = image.transform((width, int(round(height + shift_in_pixels))),
                                    Image.AFFINE,
                                    transform_matrix,
                                    Image.BICUBIC)

            image = image.crop((0, abs(shift_in_pixels), width, height))

            image = image.resize((width, height), resample=Image.BICUBIC)

        return img_to_arr(image)
