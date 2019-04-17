#
#
#   Fashion MNIST
#
#

from .model import Model
from .mixins import KerasMixin

from keras import models, layers


class FashionMnistNN(Model, KerasMixin):

    def __init__(self, classes=10):
        self.model = models.Sequential([
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(classes, activation="softmax")
        ])
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
