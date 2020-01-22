
#
#
#   Spectrogram preprocessor
#
#

import numpy as np

from scipy import signal
from ..preprocessor import Preprocessor
from ...utils.iterator_utils import apply

import logging
logger = logging.getLogger(__name__)


class Spectrogram(Preprocessor):
    """
    Audio sample spectrogram

    Parameters
    ----------
    nfft: int
        Length of each window segment
    fs: int
        Sampling frequencies
    noverlap: int
        Overlap between windows

    Note
    ----------

    nperseg: int
        By default, nfft == nperseg, meaning that no zero-padding will be used.

    """

    def __init__(self, nfft=100, fs=4000, noverlap=60):
        self.nfft = nfft
        self.fs = fs
        self.noverlap = noverlap

    def fit(self, X, y=None):
        return self

    def fit_generator(self, generator):
        return self

    def _transform_sample(self, x):
        nchannels = x.ndim
        if nchannels == 1:
            freqs, times, Sxx = signal.spectrogram(x, self.fs, noverlap=self.noverlap,
                                                   nfft=self.nfft, nperseg=self.nfft)
        elif nchannels == 2:
            freqs, times, Sxx = signal.spectrogram(x[:, 0], self.fs, noverlap=self.noverlap,
                                                   nfft=self.nfft, nperseg=self.nfft)

        # TODO: Find a better way for swapping. Perhaps another "swapper" preprocessor
        Sxx = np.swapaxes(Sxx, 1, 0)

        return Sxx

    def transform(self, X):
        return apply(X, self._transform_sample)

    def save(self, save_dir):
        pass

    def restore(self, save_dir):
        pass
