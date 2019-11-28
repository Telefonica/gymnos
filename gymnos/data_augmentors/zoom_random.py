#
#
#   Zoom random
#
#

import math
import random

from PIL import Image

from ..utils.iterator_utils import apply
from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr


class ZoomRandom(DataAugmentor):
    """
    This class is used to zoom into random areas of the image.

    Zooms into a random area of the image, rather than the centre of
    the image, as is done by :class:`Zoom`. The zoom factor is fixed
    unless :attr:`randomise` is set to ``True``.

    :param probability: Controls the probability that the operation is
     performed when it is invoked in the pipeline.
    :param percentage_area: A value between 0.1 and 1 that represents the
     area that will be cropped, with 1 meaning the entire area of the
     image will be cropped and 0.1 mean 10% of the area of the image
     will be cropped, before zooming.
    :param randomise: If ``True``, uses the :attr:`percentage_area` as an
     upper bound, and randomises the zoom level from between 0.1 and
     :attr:`percentage_area`.
    """

    def __init__(self, probability, percentage_area, randomise):
        super().__init__(probability)
        self.percentage_area = percentage_area
        self.randomise = randomise

    def transform(self, images):
        """
        Randomly zoom into the passed :attr:`image` by first cropping the image
        based on the :attr:`percentage_area` argument, and then resizing the
        image to match the size of the input area.

        Effectively, you are zooming in on random areas of the image.

        :param image: The image to crop an area from.
        :type image: np.array
        :return: The transformed image
        """
        def operation(image):
            image = arr_to_img(image)
            if self.randomise:
                r_percentage_area = round(random.uniform(0.1, self.percentage_area), 2)
            else:
                r_percentage_area = self.percentage_area

            w, h = image.size
            w_new = int(math.floor(w * r_percentage_area))
            h_new = int(math.floor(h * r_percentage_area))

            random_left_shift = random.randint(0, (w - w_new))  # Note: randint() is from uniform distribution.
            random_down_shift = random.randint(0, (h - h_new))

            image = image.crop((random_left_shift, random_down_shift, w_new + random_left_shift,
                                h_new + random_down_shift))

            image = image.resize((w, h), resample=Image.BICUBIC)
            return img_to_arr(image)

        return apply(images, operation)
