#
#
#   Crop percentage
#
#

import math
import random

from ..utils.image_utils import arr_to_img, img_to_arr
from .data_augmentor import DataAugmentor


class CropPercentage(DataAugmentor):
    """
    This class is used to crop images by a percentage of their area.
    """

    def __init__(self, probability, percentage_area, centre, randomise_percentage_area):
        """
        As well as the always required :attr:`probability` parameter, the
        constructor requires a :attr:`percentage_area` to control the area
        of the image to crop in terms of its percentage of the original image,
        and a :attr:`centre` parameter toggle whether a random area or the
        centre of the images should be cropped.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param percentage_area: The percentage area of the original image
         to crop. A value of 0.5 would crop an area that is 50% of the area
         of the original image's size.
        :param centre: Whether to crop from the centre of the image or
         crop a random location within the image.
        :type probability: Float
        :type percentage_area: Float
        :type centre: Boolean
        """
        super().__init__(probability)
        self.percentage_area = percentage_area
        self.centre = centre
        self.randomise_percentage_area = randomise_percentage_area

    def transform(self, image):
        """
        Crop the passed :attr:`image` by percentage area, returning the crop as an
        image.

        :param image: The image(s) to crop an area from.
        :type image: np.array
        :return: The transformed image
        """
        image = arr_to_img(image)
        if self.randomise_percentage_area:
            r_percentage_area = round(random.uniform(0.1, self.percentage_area), 2)
        else:
            r_percentage_area = self.percentage_area

        # The images must be of identical size, which is checked by Pipeline.ground_truth().
        w, h = image.size

        w_new = int(math.floor(w * r_percentage_area))  # TODO: Floor might return 0, so we need to check this.
        h_new = int(math.floor(h * r_percentage_area))

        left_shift = random.randint(0, int((w - w_new)))
        down_shift = random.randint(0, int((h - h_new)))

        if self.centre:
            image = image.crop(((w / 2) - (w_new / 2), (h / 2) - (h_new / 2), (w / 2) + (w_new / 2),
                                (h / 2) + (h_new / 2)))
        else:
            image = image.crop((left_shift, down_shift, w_new + left_shift, h_new + down_shift))

        return img_to_arr(image)
