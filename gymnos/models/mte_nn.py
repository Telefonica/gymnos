#
#
#   Media Tagging Engine Model
#
#

from .model import Model
from .mixins import KerasClassifierMixin
from ..utils.lazy_imports import lazy_imports
from .utils.keras_metrics import accuracy_multilabel, precision


class MTENN(KerasClassifierMixin, Model):
    """
    Task: **Multi-label classification**

    Neural network developed to solve MTE subscription classification task (:class:`gymnos.datasets.mte.MTE`).

    Can I use?
        - Generators: ✔️
        - Probability predictions: ✔️
        - Distributed datasets: ❌

    Warnings
    ------------
    This model requires labels with multi-label format e.g ``[[0, 0, 0, 1, 1, 0, 1]]``

    Parameters
    ----------
    input_shape: list
        Shape of features.
    classes: int, 17
        Number of classes to classify images into. This is useful if
        you want to train this model with another dataset.
    """

    def __init__(self, input_shape, classes=17):
        keras = lazy_imports.tensorflow.keras

        self.model = keras.models.Sequential([
            keras.layers.Dense(units=512, activation="relu", kernel_initializer="glorot_uniform"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=256, activation="relu", kernel_initializer="glorot_uniform"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=256, activation="relu", kernel_initializer="glorot_uniform"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=classes, activation="sigmoid", kernel_initializer="glorot_uniform")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[accuracy_multilabel, precision])
