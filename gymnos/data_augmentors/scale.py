#
#
#   Scale
#
#

from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr

from PIL import Image


class Scale(DataAugmentor):
    """
    This class is used to increase or decrease images in size by a certain
    factor, while maintaining the aspect ratio of the original image.

    .. seealso:: The :class:`Resize` class for resizing images by
     **dimensions**, and hence will not necessarily maintain the aspect ratio.

    This function will return images that are **larger** than the input
    images.
    """

    def __init__(self, probability, scale_factor):
        """
        As the aspect ratio is always kept constant, only a
        :attr:`scale_factor` is required for scaling the image.

        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param scale_factor: The factor by which to scale, where 1.5 would
         result in an image scaled up by 150%.
        :type probability: Float
        :type scale_factor: Float
        """
        super().__init__(probability)
        self.scale_factor = scale_factor

    def transform(self, image):
        """
        Scale the passed :attr:`image` by the factor specified during
        instantiation, returning the scaled image.

        :param image: The image to scale.
        :type image: np.array
        :return: The transformed image
        """
        image = arr_to_img(image)
        w, h = image.size

        new_h = int(h * self.scale_factor)
        new_w = int(w * self.scale_factor)

        image = image.resize((new_w, new_h), resample=Image.BICUBIC)

        return img_to_arr(image)
