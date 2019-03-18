#
#
#   Model Checkpoint
#
#

import os

from keras import callbacks


class ModelCheckpoint(callbacks.ModelCheckpoint):

    def __init__(self, save_dir, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                 mode='auto', frequency=1):
        super().__init__(
            filepath=os.path.join(save_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=frequency
        )
