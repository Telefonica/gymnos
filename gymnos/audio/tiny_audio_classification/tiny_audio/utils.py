import tensorflow_io as tfio
import librosa
import tensorflow as tf
import numpy as np
import os




#Listing wav files at root directory
def create_file_lists(classes,speech_commands_dataset_basepath):
    fullpaths = []
    targets = []
    folds = []
    count = 0
    count_train = 0
    for sounds in classes:
        directory = speech_commands_dataset_basepath + "/" + sounds
        arr = os.listdir(directory)
        count_train = 0
        for file in arr:

            fullpaths.append(directory + "/" + file)

            targets.append(count)
            if count_train >= 1400:
                folds.append(1)

            else:
                folds.append(0)

            count_train += 1
        count += 1

    return fullpaths,targets,folds



#Loading wav files with librosa
def loading_wav(filename, desired_sample_rate, desired_channels):
    try:

        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents, desired_channels=desired_channels)
        wav = tf.squeeze(wav, axis=-1)
    except:

        filename = tf.cast(filename, tf.string)
        wav, sample_rate = librosa.load(filename.numpy().decode(
            'utf-8'), sr=None, mono=(desired_channels == 1))

    wav = tfio.audio.resample(wav, rate_in=tf.cast(
        sample_rate, dtype=tf.int64), rate_out=tf.cast(desired_sample_rate, dtype=tf.int64))

    return wav

#Loading wavs
def load_wav_map(fullpath, label, fold):
    sample_rate = 16000
    channels = 1
    wav = tf.py_function(
        loading_wav, [fullpath, sample_rate, channels], tf.float32)

    return wav, label, fold


#Checking the wavs are not empty
@tf.function
def wav_not_empty(wav):
    return tf.experimental.numpy.any(wav)



#Creating Spectrogram of the raw signal
@tf.function
def creating_spectrogram(samples):
    spectrogram = tfio.audio.spectrogram(
        samples,
        nfft=256,
        window=1000,
        stride=500)

    return spectrogram


def create_spectrogram_map(samples, label, fold):
    return creating_spectrogram(samples), label, fold

def scheduler(epoch, lr):
  if epoch < 100:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


#Creating mel representation of the spectrogram
@tf.function
def creating_mel(samples):
    mel_spectrogram = tfio.audio.melscale(
        samples, rate=16000, mels=32, fmin=0, fmax=8000)

    return mel_spectrogram


def create_mel_map(samples, label, fold):
    return creating_mel(samples), label, fold
