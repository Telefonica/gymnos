#
#
#   Timing
#
#

import time


class ElapsedTimeCalculator:
    """
    Context manager to measure elapsed time.

    Attributes
    ----------
    s: float
        Elapsed time
    """

    def __init__(self):
        self.times = {}

    def __call__(self, name):
        self.current_name = name
        return self

    def __enter__(self):
        self.s = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.s = time.time() - self.s
        if self.current_name is not None:
            self.times[self.current_name] = round(self.s, ndigits=3)

        self.current_name = None
