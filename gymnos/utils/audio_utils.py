#
#
#   Audio Utils
#
#

import numpy as np
import os
import random
import sys
import io
import csv

from scipy.io import wavfile


def load_wav_file(audio_path):
    """
    Loads WAV audio file

    Parameters
    ----------
    audio_path: str
        Path of the audio.

    Returns
    -------
    rate: int
        Sampling frequency of the audio file.
    data: np.array
        Array of audio samples.
    """
    rate, data = wavfile.read(audio_path)
    return rate, data