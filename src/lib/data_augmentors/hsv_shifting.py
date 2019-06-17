#
#
#   HSV Shifting
#
#

import numpy as np

from PIL import Image

from .data_augmentor import DataAugmentor
from ..utils.image_utils import arr_to_img, img_to_arr


class HSVShifting(DataAugmentor):
    """
    CURRENTLY NOT IMPLEMENTED.
    """

    def __init__(self, probability, hue_shift, saturation_scale, saturation_shift, value_scale, value_shift):
        super().__init__(probability)
        self.hue_shift = hue_shift
        self.saturation_scale = saturation_scale
        self.saturation_shift = saturation_shift
        self.value_scale = value_scale
        self.value_shift = value_shift

    def transform(self, image):
        image = arr_to_img(image)
        hsv = np.array(image.convert("HSV"), 'float64')
        hsv /= 255.

        hsv[..., 0] += np.random.uniform(-self.hue_shift, self.hue_shift)
        hsv[..., 1] *= np.random.uniform(1 / (1 + self.saturation_scale), 1 + self.saturation_scale)
        hsv[..., 1] += np.random.uniform(-self.saturation_shift, self.saturation_shift)
        hsv[..., 2] *= np.random.uniform(1 / (1 + self.value_scale), 1 + self.value_scale)
        hsv[..., 2] += np.random.uniform(-self.value_shift, self.value_shift)

        hsv.clip(0, 1, hsv)
        hsv = np.uint8(np.round(hsv * 255.))

        image = Image.fromarray(hsv, "HSV").convert("RGB")

        return img_to_arr(image)
