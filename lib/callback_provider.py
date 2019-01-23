import os, logging, json
import numpy as np
import tensorflow as tf

from callbacks.reduce_learning import *
from callbacks.early_stopping import *
from callbacks.model_checkpoint import *
from callbacks.tensorboard import *

from var.callbacks import *

class CallbackProvider(object):
    def __init__(self, config):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "CALLBACK_PROVIDER"
        self._config = config
        self._callbackList = []

    def getList(self):
        return self._callbackList

    def buildCallbackList(self, runTimeConfig):
        self._runTimeConfig = runTimeConfig
        self._log.debug("{0} - Building callback list ...".format(self._log_prefix))
        for callback in self._config["list"]: 
            self._callbackList.append(self.__factory(callback["id"], callback["options"]).getInstance())

    def __factory(self, callbackId, config):
        cbInstance = None
        if callbackId == REDUCE_LEARNING: cbInstance = ReduceLearning(config)
        elif callbackId == EARLY_STOPPING: cbInstance = EarlyStopping(config)
        elif callbackId == MODEL_CHECKPOINT: cbInstance = ModelCheckpoint(config, self._runTimeConfig)
        elif callbackId == TENSORBOARD: cbInstance = TensorBoard(config, self._runTimeConfig)   
        else:
            errMsg = "{0} - Callback suppport for {1} not available.".format(self._log_prefix, callbackId)
            self._log.error(errMsg)
            raise ValueError(errMsg)
        
        if cbInstance:
            #self._log.debug("{0} - Instantiating {1} callback ...".format(self._log_prefix, callbackId))
            return cbInstance