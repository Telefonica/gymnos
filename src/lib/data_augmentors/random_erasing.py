#
#
# Random Erasing
#
#

import random
import numpy as np

from PIL import Image

from .data_augmentor import DataAugmentor


class RandomErasing(DataAugmentor):
    """
    Class that performs Random Erasing, an augmentation technique described
    in `https://arxiv.org/abs/1708.04896 <https://arxiv.org/abs/1708.04896>`_
    by Zhong et al. To quote the authors, random erasing:

    "*... randomly selects a rectangle region in an image, and erases its
    pixels with random values.*"

    Exactly this is provided by this class.

    Random Erasing can make a trained neural network more robust to occlusion.
    """

    def __init__(self, probability, rectangle_area):
        """
        The size of the random rectangle is controlled using the
        :attr:`rectangle_area` parameter. This area is random in its
        width and height.

        :param probability: The probability that the operation will be
         performed.
        :param rectangle_area: The percentage are of the image to occlude.
        """
        super().__init__(probability)
        self.rectangle_area = rectangle_area

    def transform(self, image):
        """
        Adds a random noise rectangle to a random area of the passed image,
        returning the original image with this rectangle superimposed.

        :param image: The image(s) to add a random noise rectangle to.
        :type image: np.array
        :return: The transformed image
        """
        image = Image.fromarray(image)
        w, h = image.size

        w_occlusion_max = int(w * self.rectangle_area)
        h_occlusion_max = int(h * self.rectangle_area)

        w_occlusion_min = int(w * 0.1)
        h_occlusion_min = int(h * 0.1)

        w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

        if len(image.getbands()) == 1:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
        else:
            rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion,
                                                                len(image.getbands())) * 255))

        random_position_x = random.randint(0, w - w_occlusion)
        random_position_y = random.randint(0, h - h_occlusion)

        image.paste(rectangle, (random_position_x, random_position_y))

        return np.array(image)
