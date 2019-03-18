#
#
#   Reduce Learning
#
#

from keras import callbacks


class ReduceLearning(callbacks.ReduceLROnPlateau):

    def __init__(self, monitor="val_loss", factor=0.1, patience=10, verbose=0, mode="auto",
                 min_delta=1e-4, cooldown=0, min_lr=0):
        super().__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr
        )
