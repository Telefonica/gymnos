#
#
#   Image Utils
#
#

import numpy as np

from PIL import Image


def imread_rgb(image_path):
    """
    Read RGB image

    Parameters
    ----------
    image_path: str
        Path of the image.

    Returns
    -------
    image: np.array
        Array of pixels.
    """
    img = Image.open(image_path)
    return img_to_arr(img.convert("RGB"))


def arr_to_img(arr):
    return Image.fromarray(arr.squeeze().astype(np.uint8))


def img_to_arr(img):
    arr = np.array(img, dtype=np.uint8)

    if arr.ndim < 3:
        arr = arr[..., np.newaxis]

    return arr


def imgray(rgb_arr):
    """
    Convert image to grayscale

    Parameters
    ----------
    rgb_arr: np.array
        3D array of pixels

    Returns
    -------
    gray_img: np.array
        Grasycale array.
    """
    img = arr_to_img(rgb_arr)
    img = img.convert("L")
    return img_to_arr(img)


def imresize(rgb_arr, size):
    """
    Resize image to given ``size``.

    Parameters
    ----------
    rgb_arr: np.array
        3D array of pixels.
    size: tuple or int
        Width and height for the new image

    Returns
    -------
    resized_img: np.array
        Resized image.
    """
    img = arr_to_img(rgb_arr)
    if isinstance(size, (list, tuple)):
        img = img.resize(size, Image.ANTIALIAS)
    else:
        img = img.resize((size, size), Image.ANTIALIAS)

    return img_to_arr(img)
