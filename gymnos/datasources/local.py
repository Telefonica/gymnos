#
#
#   Local data source
#
#

import os

from hydra.utils import get_original_cwd


def LocalDataSource(path):
    if not os.path.isabs(path):
        try:
            cwd = get_original_cwd()
        except ValueError:
            cwd = os.getcwd()

        path = os.path.join(cwd, path)

    return path
