#
#
#   Media Tagging Engine Model
#
#

from tensorflow.keras import models, layers

from .model import Model
from .mixins import KerasClassifierMixin
from .utils.keras_metrics import accuracy_multilabel, precision


class MTENN(KerasClassifierMixin, Model):
    """
    Neural network developed to solve MTE subscription classification task (:class:`gymnos.datasets.mte.MTE`).

    Parameters
    ----------
    input_shape: list
        Shape of features.
    classes: int, 17
        Number of classes to classify images into. This is useful if
        you want to train this model with another dataset.

    Note
    ----
    This model requires labels with multi-label format.
    """

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
