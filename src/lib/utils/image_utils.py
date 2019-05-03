#
#
#   Image Utils
#
#

import cv2 as cv


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
    bgr_image = cv.imread(image_path)
    return cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)


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
    return cv.cvtColor(rgb_arr, cv.COLOR_RGB2GRAY)


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
    return cv.resize(rgb_arr, size)
