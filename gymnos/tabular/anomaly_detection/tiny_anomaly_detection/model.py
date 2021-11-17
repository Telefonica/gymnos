

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
tf.random.set_seed(6)


def create_tiny_model(encoding_dim, dropout, sample_shape):

    # Creating the keras model
    model = models.Sequential([
        layers.InputLayer(input_shape=(30,)),
        layers.Dense(30, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(15, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(5 , activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(5 , activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(15, activation='relu'),
        layers.Dense(*sample_shape, activation='relu')
    ])

    return model
