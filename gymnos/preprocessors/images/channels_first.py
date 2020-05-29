#
#
#   Channels first
#
#

import numpy as np

from ..preprocessor import Preprocessor


class ChannelsFirst(Preprocessor):
    """
    Change image order from height x width x channels to channel x height x width
    """

    def transform(self, X):
        assert np.ndim(X) == 4  # batch_size, height, width, channels
        return np.rollaxis(X, -1, 1)
