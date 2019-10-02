#
#
#   CNN
#
#

from tensorflow.keras import models, layers, optimizers

from .model import Model
from .mixins import KerasClassifierMixin


class DogsVsCatsCNN(KerasClassifierMixin, Model):
    """
    Convolutional neuronal network developed to solve Dogs vs Cats image classification
    task (:class:`gymnos.datasets.dogs_vs_cats.DogsVsCats`).

    Parameters
    ----------
    input_shape: list
        Data shape expected.
    classes: int, optional
        Optional number of classes to classify images into. This is useful if
        you want to train this model with another dataset.

    Note
    ----
    This model requires one-hot encoded labels.

    Examples
    --------
    .. code-block:: py

        DogsVsCatsCNN(
            input_shape=[120, 120, 3],
            classes=2
        )
    """

    def __init__(self, input_shape, classes=2):
        self.model = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=[3, 3], activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2]),
            layers.Dropout(rate=0.7),
            layers.Conv2D(filters=64, kernel_size=[3, 3], activation="relu"),
            layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2]),
            layers.Dropout(rate=0.7),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(rate=0.5),
            layers.Dense(classes, activation="softmax")
        ])

        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy",
                           metrics=["accuracy"])
