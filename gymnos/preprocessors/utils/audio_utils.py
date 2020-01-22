#
#
#   Audio Utils
#
#

from ...utils.lazy_imports import lazy_imports as lazy


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

    scipy = __import__("{}.io.wavfile".format(lazy.scipy.__name__))
    rate, data = scipy.io.wavfile.read(audio_path)
    return rate, data
