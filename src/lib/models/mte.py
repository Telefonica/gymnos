#
#
#   Media Tagging Engine Model
#
#

from keras import models, layers

from .model import Model
from .mixins import KerasMixin

from ..utils.keras_metrics import accuracy_multilabel, precision


class MTE(Model, KerasMixin):

    def __init__(self, input_shape, classes=17):
        self.model = models.Sequential([
            layers.Dense(units=512, activation="relu", kernel_initializer="glorot_uniform"),
            layers.Dropout(0.5),
            layers.Dense(units=256, activation="relu", kernel_initializer="glorot_uniform"),
            layers.Dropout(0.5),
            layers.Dense(units=256, activation="relu", kernel_initializer="glorot_uniform"),
            layers.Dropout(0.5),
            layers.Dense(units=classes, activation="sigmoid", kernel_initializer="glorot_uniform")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[accuracy_multilabel, precision])
