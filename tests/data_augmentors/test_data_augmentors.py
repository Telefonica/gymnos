#
#
#   Test common
#
#


import pytest
import numpy as np
from gymnos.data_augmentors import Distort, Flip, GaussianDistortion, Greyscale, HistogramEqualisation, \
    Invert, RandomBrightness, RandomColor, RandomContrast, RandomErasing, Rotate, RotateRange, \
    Shear, Skew, Zoom, ZoomGroundTruth, ZoomRandom


@pytest.mark.parametrize(("da_instance"), [
    Distort(probability=1, grid_width=20, grid_height=20, magnitude=10),
    Flip(probability=1, top_bottom_left_right="LEFT_RIGHT"),
    GaussianDistortion(probability=1, grid_width=30, grid_height=50, magnitude=10, corner="bell", method="in",
                       mex=2.0, mey=0.5, sdx=5.2, sdy=3.5),
    Greyscale(probability=1),
    HistogramEqualisation(probability=1),
    Invert(probability=1),
    RandomBrightness(probability=1, min_factor=2.5, max_factor=5.6),
    RandomColor(probability=1, min_factor=0.5, max_factor=0.75),
    RandomContrast(probability=1, min_factor=0.5, max_factor=0.8),
    RandomErasing(probability=1, rectangle_area=0.5),
    Rotate(probability=1, rotation=90),
    RotateRange(probability=1, max_left_rotation=120, max_right_rotation=30),
    Shear(probability=1, max_shear_left=2, max_shear_right=4),
    Skew(probability=1, skew_type="TILT", magnitude=35),
    Zoom(probability=1, min_factor=5.0, max_factor=20.0),
    ZoomGroundTruth(probability=1, min_factor=5.0, max_factor=20.0),
    ZoomRandom(probability=1, percentage_area=0.5, randomise=True)
])
def test_transform(da_instance):
    for size in [(50, 50, 3), (50, 50, 1)]:
        image = np.random.randint(0, 255, size, dtype=np.uint8)
        new_image = da_instance.transform(image)

        assert new_image.shape == size

        if da_instance.__class__.__name__ in ("Greyscale", "RandomColor") and size == (50, 50, 1):
            assert np.array_equal(image, new_image)
        else:
            assert not np.array_equal(image, new_image)
