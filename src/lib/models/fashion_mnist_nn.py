#
#
#   Fashion MNIST
#
#

from .model import KerasModel

from keras import models, layers


class FashionMnistNN(KerasModel):

    def __init__(self, input_shape, **hyperparameters):
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])

        super().__init__(input_shape, model, **hyperparameters)
