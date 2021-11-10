import tensorflow_io as tfio
import librosa
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot


#Listing wav files at root directory
def create_file_lists():
    fullpaths = []
    targets = []
    folds = []
    for sounds in classes:
        directory = speech_commands_dataset_basepath + "/" + sounds
        arr = os.listdir(directory)
        for file in arr:
            fullpaths.append(directory + "/" + file)
            targets.append(count)
            if count_train >= 35:
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


def load_wav_map(fullpath, label, fold):
    sample_rate = 16000
    channels = 1
    wav = tf.py_function(
        loading_wav, [fullpath, sample_rate, channels], tf.float32)

    return wav, label, fold


@tf.function
def splitting_wav(wav, width, stride):
    return tf.map_fn(fn=lambda t: wav[t * stride:t * stride + width], elems=tf.range((tf.shape(wav)[0] - width) // stride), fn_output_signature=tf.float32)


@tf.function
def wav_not_empty(wav):
    return tf.experimental.numpy.any(wav)


def split_wav_map(wav, label, fold):
    wavs = splitting_wav(wav, width=16000, stride=4000)
    labels = tf.repeat(label, tf.shape(wavs)[0])
    folds = tf.repeat(fold, tf.shape(wavs)[0])

    return tf.data.Dataset.from_tensor_slices((wavs, labels, folds))


@tf.function
def creating_spectrogram(samples):
    return tf.abs(
        tf.signal.stft(samples, frame_length=256, frame_step=128)
    )


def create_spectrogram_map(samples, label, fold):
    return creating_spectrogram(samples), label, fold

def scheduler(epoch, lr):
  if epoch < 100:
    return lr
  else:
    return lr * tf.math.exp(-0.1)



def apply_qat_to_dense_and_cnn(layer):
  if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer
