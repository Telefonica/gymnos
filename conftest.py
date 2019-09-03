#
#
#   Test configuration
#
#

import pytest
import numpy as np


def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", default=False, help="run integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        # --integration given in cli: do not skip integration tests
        return

    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def random_rgb_image():
    return np.random.randint(0, 255, [100, 100, 3], dtype=np.uint8)


@pytest.fixture
def random_gray_image():
    return np.random.randint(0, 255, [100, 100, 1], dtype=np.uint8)


@pytest.fixture(scope='session')
def session_tmp_path(tmpdir_factory):
    return tmpdir_factory.mktemp("gymnos")
