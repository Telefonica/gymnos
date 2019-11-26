import gymnos
import pytest
import numpy as np

from gymnos.data_augmentors.distort import Distort
from gymnos.data_augmentors.flip import Flip
from gymnos.data_augmentors.gaussian_distortion import GaussianDistortion
from gymnos.data_augmentors.grayscale import Grayscale
from gymnos.data_augmentors.histogram_equalisation import HistogramEqualisation
from gymnos.data_augmentors.invert import Invert
from gymnos.data_augmentors.random_brightness import RandomBrightness
from gymnos.data_augmentors.random_color import RandomColor
from gymnos.data_augmentors.random_contrast import RandomContrast
from gymnos.data_augmentors.random_erasing import RandomErasing
from gymnos.data_augmentors.rotate import Rotate
from gymnos.data_augmentors.shear import Shear
from gymnos.data_augmentors.skew import Skew
from gymnos.data_augmentors.zoom import Zoom
from gymnos.data_augmentors.zoom_ground_truth import ZoomGroundTruth
from gymnos.data_augmentors.zoom_random import ZoomRandom


def test_load():
    data_augmentor = gymnos.data_augmentors.load("invert", probability=0.5)

    assert isinstance(data_augmentor, gymnos.data_augmentors.data_augmentor.DataAugmentor)

    with pytest.raises(ValueError):
        _ = gymnos.data_augmentors.load("dummy")


@pytest.mark.parametrize(("da_instance"), [
    Distort(probability=1, grid_width=20, grid_height=20, magnitude=10),
    Flip(probability=1, top_bottom_left_right="LEFT_RIGHT"),
    GaussianDistortion(probability=1, grid_width=30, grid_height=50, magnitude=10,
                       corner="bell", method="in", mex=2.0, mey=0.5, sdx=5.2, sdy=3.5),
    Grayscale(probability=1),
    HistogramEqualisation(probability=1),
    Invert(probability=1),
    RandomBrightness(probability=1, min_factor=2.5, max_factor=5.6),
    RandomColor(probability=1, min_factor=0.5, max_factor=0.75),
    RandomContrast(probability=1, min_factor=0.5, max_factor=0.8),
    RandomErasing(probability=1, rectangle_area=0.5),
    Rotate(probability=1, rotation=90),
    Shear(probability=1, max_shear_left=2, max_shear_right=4),
    Skew(probability=1, skew_type="TILT", magnitude=35),
    Zoom(probability=1, min_factor=5.0, max_factor=20.0),
    ZoomGroundTruth(probability=1, min_factor=5.0, max_factor=20.0),
    ZoomRandom(probability=1, percentage_area=0.5, randomise=True)
])
def test_transform(da_instance):
    for size in [(1, 50, 50, 3), (1, 50, 50, 1)]:
        images = np.random.randint(0, 255, size, dtype=np.uint8)
        new_images = da_instance.transform(images)

        assert new_images.shape == size

        if da_instance.__class__.__name__ in ("Grayscale", "RandomColor") and size == (1, 50, 50, 1):
            assert np.array_equal(images, new_images)
        else:
            assert not np.array_equal(images, new_images)
