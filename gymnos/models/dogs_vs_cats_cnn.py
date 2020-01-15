#
#
#   CNN
#
#

from .model import Model
from .mixins import KerasClassifierMixin
from ..utils.lazy_imports import lazy_imports


class DogsVsCatsCNN(KerasClassifierMixin, Model):
    """
    Task: **Classification**

    Convolutional neuronal network developed to solve Dogs vs Cats image classification
    task (:class:`gymnos.datasets.dogs_vs_cats.DogsVsCats`)

    Can I use?
        - Generators: ✔️
        - Probability predictions: ✔️
        - Distributed datasets: ❌

    Parameters
    ----------
    input_shape: list
        Data shape expected.
    classes: int, optional
        Optional number of classes to classify images into. This is useful if
        you want to train this model with another dataset.

    Warnings
    ----------------
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
        keras = lazy_imports.tensorflow.keras

        self.model = keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=[3, 3], activation="relu", input_shape=input_shape),
            keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2]),
            keras.layers.Dropout(rate=0.7),
            keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation="relu"),
            keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2]),
            keras.layers.Dropout(rate=0.7),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(classes, activation="softmax")
        ])

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy",
                           metrics=["accuracy"])
