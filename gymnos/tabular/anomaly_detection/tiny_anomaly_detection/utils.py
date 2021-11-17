from os import listdir
from os.path import join
import numpy as np
import scipy
from scipy import stats
import logging
import tensorflow as tf
from .c_writer import create_array, create_header


def createFilenameList(root, op_list):
    # Creating a list of csv files which corresponds with the samples
    op_filenames = []
    num_samples = 0
    for index, target in enumerate(op_list):
        samples_in_dir = listdir(join(root, target))
        samples_in_dir = [join(root, target, sample)
                          for sample in samples_in_dir]
        op_filenames.append(samples_in_dir)

    return [item for sublist in op_filenames for item in sublist]


def extract_features(sample, max_measurements=0, scale=1):
    # For each of the 15 input samples, we calculate the MAD.
    features = []

    if max_measurements == 0:
        max_measurements = sample.shape[0]
    sample = sample[0:max_measurements]

    # The 15 input sample ends in a single value
    features.append(stats.median_abs_deviation(sample))
    return np.array(features).flatten()


def create_feature_set(filenames, max_measurements):
    x_out = []
    # For each of the samples files, we extract the MAD
    number_files = 0
    partial_features = []
    for file in filenames:

        sample = np.genfromtxt(file, delimiter=',')
        features = extract_features(sample, max_measurements)
        partial_features.append(features)
        number_files += 1
        if number_files == 10:
            features = np.array(partial_features).reshape(1, 30)
            partial_features = []
            x_out.append(features)
            number_files = 0


    return np.array(x_out)


def convert_to_tiny(self):

    c_model_name = "tiny_autoencoder"
    logger = logging.getLogger(__name__)
    logger.info(
        "The following steps are needed if running in MCUs is wanted")

    # Converting model to TF lite
    converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
    self.tflite_model = converter.convert()

    open(join(self.directory + "/", self.tflite_model_name) +
         '.tflite', 'wb').write(self.tflite_model)
    # Adapting the model to run on C
    hex_array = [format(val, '#04x') for val in self.tflite_model]
    c_model = create_array(
        np.array(hex_array), 'unsigned char', "micro-model")

    header_str = create_header(c_model, c_model_name)
    logger.info("Model ready! ")
    logger.info("------------ ")
    with open(join(self.directory + "/", c_model_name) + '.h', 'w') as file:
        file.write(header_str)
