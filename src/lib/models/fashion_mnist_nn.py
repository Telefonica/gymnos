#
#
#   Fashion MNIST
#
#

from .model import Model
from .mixins import KerasMixin

from keras import models, layers


class FashionMnistNN(Model, KerasMixin):
    """
    Neural network developed to solve Fashion MNIST image classification task.

    Note
    ----
    This model can be useful to see the development of a Keras model on the platform.

    Parameters
    ----------
    classes: int, optional
        Number of classes to classify images into. This is useful if
        you want to train this model with another dataset.
    """

    def __init__(self, classes=10):
        self.model = models.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(classes, activation="softmax")
        ])
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
