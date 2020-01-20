#
#
#   Test image utils
#
#

import numpy as np

from gymnos.utils.lazy_imports import lazy_imports as lazy
from gymnos.preprocessors.utils.image_ops import arr_to_img, img_to_arr, imgray, imresize


def test_arr_to_img():
    image = np.random.randint(0, 255, [100, 100, 3], dtype=np.uint8)
    new_image = arr_to_img(image)
    assert isinstance(new_image, lazy.PIL.Image.Image)
    assert np.array(new_image).ndim == 3

    image = np.random.randint(0, 255, [100, 100, 1], dtype=np.uint8)
    new_image = arr_to_img(image)
    assert isinstance(new_image, lazy.PIL.Image.Image)
    assert np.array(new_image).ndim == 2

    image = np.random.randint(0, 255, [100, 100], dtype=np.uint8)
    new_image = arr_to_img(image)
    assert isinstance(new_image, lazy.PIL.Image.Image)
    assert np.array(new_image).ndim == 2


def test_img_to_arr(random_rgb_image, random_gray_image):
    image = lazy.PIL.Image.fromarray(random_rgb_image)
    arr_image = img_to_arr(image)
    assert arr_image.shape[-1] == 3

    image = lazy.PIL.Image.fromarray(np.squeeze(random_gray_image))
    arr_image = img_to_arr(image)
    assert arr_image.shape[-1] == 1


def test_imgray(random_rgb_image):
    new_image = imgray(random_rgb_image)
    assert random_rgb_image.shape[-1] == 3
    assert new_image.shape[-1] == 1


def test_imresize(random_rgb_image, random_gray_image):
    assert random_rgb_image.shape != (150, 150, 3)
    new_image = imresize(random_rgb_image, [150, 150])
    assert new_image.shape == (150, 150, 3)
