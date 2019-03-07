import logging

from . import callback

from keras import callbacks

class EarlyStopping(callback.Callback):  
    def __init__(self, config):
        callback.Callback.__init__(self)
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "EARLY_STOPPING"
        self._config = config
        self._monitor = config["monitor"] if "monitor" in config else'val_loss'
        self._min_delta = config["min_delta"] if "min_delta" in config else 0
        self._patience = config["patience"] if "patience" in config else 0
        self._verbose = config["verbose"] if "verbose" in config else 0
        self._mode = config["mode"] if "mode" in config else'auto'
        self._baseline = config["baseline"] if "baseline" in config else None
        self._restore_best_weights = config["restore_best_weights"] if "restore_best_weights" in config else False 
        self.__buildCallback()

    def getInstance(self):
        return self._instance

    def __buildCallback(self):
        # Only Keras support so far
        self._log.debug("{0} - Instance with params:\n[\n\t - monitor = {1}\
                                                    \n\t - min_delta = {2}\
                                                    \n\t - patience = {3}\
                                                    \n\t - verbose = {4}\
                                                    \n\t - mode = {5}\
                                                    \n\t - baseline = {6}\
                                                    \n\t - restore_best_weights = {7}\
                                                    \n]".format( self._log_prefix,
                                                                 self._monitor,
                                                                 self._min_delta, 
                                                                 self._patience, 
                                                                 self._verbose, 
                                                                 self._mode, 
                                                                 self._baseline, 
                                                                 self._restore_best_weights ) )
        self._instance = callbacks.EarlyStopping( monitor=self._monitor, 
                                                  min_delta=self._min_delta, 
                                                  patience=self._patience, 
                                                  verbose=self._verbose, 
                                                  mode=self._mode, 
                                                  baseline=self._baseline, 
                                                  restore_best_weights=self._restore_best_weights )
                                               
