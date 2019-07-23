#
#
#   Test configuration
#
#

import pytest
import numpy as np


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def random_rgb_image():
    return np.random.randint(0, 255, [100, 100, 3], dtype=np.uint8)


@pytest.fixture
def random_gray_image():
    return np.random.randint(0, 255, [100, 100, 1], dtype=np.uint8)
