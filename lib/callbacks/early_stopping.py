#
#
#   Early Stopping
#
#

from keras import callbacks


class EarlyStopping(callbacks.EarlyStopping):

    def __init__(self, monitor="val_loss", min_delta=0, patience=0, verbose=0, mode="auto",
                 baseline=None, restore_best_weights=False):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights
        )
