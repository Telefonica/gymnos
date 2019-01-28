import os, logging
import callback

from keras import callbacks
from var.system_paths import *

class ModelCheckpoint(callback.Callback):  
    def __init__(self, config, runTimeConfig):
        callback.Callback.__init__(self)
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "MODEL_CHECKPOINT"
        self._config = config
        self._runTimeConfig = runTimeConfig
        self._filename = config["filename"] if "filename" in config else 'weights.\{epoch:02d\}.hdf5'
        self._log_dir = "{0}/{1}".format(runTimeConfig["train_dir"], self._filename) if "train_dir" in runTimeConfig else './logs'
        self._monitor = config["monitor"] if "monitor" in config else 'val_loss'
        self._verbose = config["verbose"] if "verbose" in config else 0
        self._save_best_only = config["save_best_only"] if "save_best_only" in config else False
        self._save_weights_only = config["ave_weights_only"] if "ave_weights_only" in config else False
        self._mode = config["mode"] if "mode" in config else "auto"
        self._period = config["period"] if "period" in config else 1
        self.__buildCallback()

    def getInstance(self):
        return self._instance

    def __buildCallback(self):
        # Only Keras support so far
        self._log.debug("{0} - Instance with params:\n[\n\t - log_dir = {1}\
                                                    \n\t - monitor = {2}\
                                                    \n\t - verbose = {3}\
                                                    \n\t - save_best_only = {4}\
                                                    \n\t - save_weights_only = {5}\
                                                    \n\t - mode = {6}\
                                                    \n\t - period = {7}\
                                                    \n]".format( self._log_prefix,
                                                                 self._log_dir,
                                                                 self._monitor, 
                                                                 self._verbose, 
                                                                 self._save_best_only, 
                                                                 self._save_weights_only, 
                                                                 self._mode, 
                                                                 self._period ) )
        self._instance = callbacks.ModelCheckpoint( self._log_dir, 
                                                    monitor=self._monitor, 
                                                    verbose=self._verbose, 
                                                    save_best_only=self._save_best_only, 
                                                    save_weights_only=self._save_weights_only, 
                                                    mode=self._mode, 
                                                    period=self._period )
                                                   
