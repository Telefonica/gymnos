#
#
#   Path
#
#

import os

from contextlib import contextmanager


@contextmanager
def chdir(path):
    """
    Context manager to temporarily change the working directory

    Parameters
    ----------
    path: str
        New working directory
    """
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)
