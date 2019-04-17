#
#
#   Path
#
#

import os

from contextlib import contextmanager


@contextmanager
def chdir(path):
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)
