


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
tf.random.set_seed(6)


def create_tiny_model(encoding_dim,dropout):

    model = models.Sequential([
    layers.InputLayer(input_shape=(3,)),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(encoding_dim-1, activation='relu'),
    layers.Dropout(dropout),
    layers.Dense(encoding_dim-1, activation='relu'),
    layers.Dense(*sample_shape, activation='relu')
    ])

    return model
