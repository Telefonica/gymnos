#
#
#   Crop
#
#

import random

from ..utils.image_utils import arr_to_img, img_to_arr
from .data_augmentor import DataAugmentor


class Crop(DataAugmentor):
    """
    This class is used to crop images by absolute values passed as parameters.
    """

    def __init__(self, probability, width, height, centre):
        """
        As well as the always required :attr:`probability` parameter, the
        constructor requires a :attr:`width` to control the width of
        of the area to crop as well as a :attr:`height` parameter
        to control the height of the area to crop. Also, whether the
        area to crop should be taken from the centre of the image or from a
        random location within the image is toggled using :attr:`centre`.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param width: The width in pixels of the area to crop from the image.
        :param height: The height in pixels of the area to crop from the image.
        :param centre: Whether to crop from the centre of the image or a random
         location within the image, while maintaining the size of the crop
         without cropping out of the original image's area.
        :type probability: Float
        :type width: Integer
        :type height: Integer
        :type centre: Boolean
        """
        super().__init__(probability)
        self.width = width
        self.height = height
        self.centre = centre

    def transform(self, image):
        """
        Crop an area from an image, either from a random location or centred,
        using the dimensions supplied during instantiation.

        :param image: The image(s) to crop the area from.
        :type image: np.array
        :return: The transformed image
        """
        image = arr_to_img(image)
        w, h = image.size  # All images must be the same size, so we can just check the first image in the list

        left_shift = random.randint(0, int((w - self.width)))
        down_shift = random.randint(0, int((h - self.height)))

        # TODO: Fix. We may want a full crop.
        if self.width > w or self.height > h:
            return img_to_arr(image)

        if self.centre:
            image = image.crop(((w / 2) - (self.width / 2), (h / 2) - (self.height / 2), (w / 2) + (self.width / 2),
                                (h / 2) + (self.height / 2)))
        else:
            image = image.crop((left_shift, down_shift, self.width + left_shift, self.height + down_shift))

        return img_to_arr(image)
