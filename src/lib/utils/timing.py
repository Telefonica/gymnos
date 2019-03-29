#
#
#   Timing
#
#

import time


class elapsed_time:

    def __enter__(self):
        self.s = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.s = time.time() - self.s
