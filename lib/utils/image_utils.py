#
#
#   Image Utils
#
#

import cv2 as cv


def imread_rgb(image_path):
    bgr_image = cv.imread(image_path)
    return cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)


def imgray(rgb_arr):
    return cv.cvtColor(rgb_arr, cv.COLOR_RGB2GRAY)


def imresize(rgb_arr, size):
    return cv.resize(rgb_arr, size)
