
#
#
#   TriggerWordDetection
#
#

import logging

from .model import Model
from .mixins import KerasClassifierMixin
from ..utils.lazy_imports import lazy_imports

logger = logging.getLogger(__name__)


class TriggerWordDetection(KerasClassifierMixin, Model):
    """
    A keras based model for trigger word detection (sometimes also called keyword detection, or wakeword detection)
    Trigger word detection is the technology that allows devices to wake up upon hearing a certain word.
    ref: https://github.com/Tony607/Keras-Trigger-Word/blob/master/Trigger%20word%20detection%20-%20v1.ipynb

    Parameters
    ----------
    input_shape: list
        shape of the model's input data (using Keras conventions).

    Examples
    --------
    .. code-block:: py

        TriggerWordDetection(
            input_shape=[5511, 101]
        )
    """

    def __init__(self, input_shape):
        keras = lazy_imports.tensorflow.keras

        self.model = keras.models.Sequential([
            keras.layers.Conv1D(196, kernel_size=15, strides=4, input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.8),
            keras.layers.GRU(units=128, return_sequences=True),
            keras.layers.Dropout(0.8),
            keras.layers.BatchNormalization(),
            keras.layers.GRU(units=128, return_sequences=True),
            keras.layers.Dropout(0.8),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.8),
            keras.layers.TimeDistributed(keras.layers.Dense(1, activation="sigmoid"))
        ])
        opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        logger.info("Model Summary:")
        logger.info(self.model.summary())
