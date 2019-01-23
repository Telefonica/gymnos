import os, logging
import callback

from keras import callbacks

class ReduceLearning(callback.Callback):  
    def __init__(self, config):
        callback.Callback.__init__(self)
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "REDUCE_LEARNING"
        self._config = config
        self._monitor = config["monitor"] if "monitor" in config else 'val_loss'
        self._factor = config["factor"] if "factor" in config else 0.1
        self._patience = config["patience"] if "patience" in config else 10 
        self._verbose = config["verbose"] if "verbose" in config else 0 
        self._mode = config["mode "] if "mode " in config else 'auto'
        self._min_delta = config["min_delta"] if "min_delta" in config else 0.0001
        self._cooldown = config["cooldown"] if "cooldown" in config else 0 
        self._min_lr = config["min_lr"] if "min_lr" in config else 0 
        self.__buildCallback()

    def getInstance(self):
        return self._instance

    def __buildCallback(self):
        # Only Keras support so far
        self._log.debug("{0} - Instance with params:\n[\n\t - monitor = {1}\
                                                    \n\t - factor = {2}\
                                                    \n\t - patience = {3}\
                                                    \n\t - verbose = {4}\
                                                    \n\t - mode = {5}\
                                                    \n\t - min_delta = {6}\
                                                    \n\t - cooldown = {7}\
                                                    \n\t - min_lr = {8}\
                                                    \n]".format( self._log_prefix,
                                                                 self._monitor, 
                                                                 self._factor, 
                                                                 self._patience, 
                                                                 self._verbose, 
                                                                 self._mode, 
                                                                 self._min_delta, 
                                                                 self._cooldown, 
                                                                 self._min_lr ) )

        self._instance = callbacks.ReduceLROnPlateau( monitor=self._monitor, 
                                                      factor=self._factor, 
                                                      patience=self._patience, 
                                                      verbose=self._verbose, 
                                                      mode=self._mode, 
                                                      min_delta=self._min_delta, 
                                                      cooldown=self._cooldown, 
                                                      min_lr=self._min_lr )

    