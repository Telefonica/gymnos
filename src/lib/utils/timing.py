#
#
#   Timing
#
#

import time


class elapsed_time:
    """
    Context manager to measure elapsed time.

    Attributes
    ----------
    s: float
        Elapsed time
    """

    def __enter__(self):
        self.s = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.s = time.time() - self.s
