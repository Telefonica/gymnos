from .callbacks.early_stopping import EarlyStopping
from .callbacks.model_checkpoint import ModelCheckpoint
from .callbacks.reduce_learning import ReduceLearning
from .callbacks.tensorboard import TensorBoard
from .log import logger
from .var.callbacks import REDUCE_LEARNING, EARLY_STOPPING, MODEL_CHECKPOINT, TENSORBOARD


class CallbackProvider(object):
    def __init__(self, config):
        self._log = logger.get_logger()
        self._log_prefix = logger.setup_prefix(__class__)
        self._config = config
        self._callbackList = []

    def getList(self):
        return self._callbackList

    def buildCallbackList(self, runTimeConfig):
        self._runTimeConfig = runTimeConfig
        self._log.debug("{0} - Building callback list ...".format(self._log_prefix))
        for callback in self._config["list"]:
            self._callbackList.append(self.__factory(callback["id"], callback["options"]).getInstance())

    def isCallbackPresentInList(self, callbackId):
        retVal = False
        for callback in self._config["list"]:
            if callback["id"] == callbackId:
                return True
        return retVal

    def __factory(self, callbackId, config):
        cbInstance = None
        if callbackId == REDUCE_LEARNING:
            cbInstance = ReduceLearning(config)
        elif callbackId == EARLY_STOPPING:
            cbInstance = EarlyStopping(config)
        elif callbackId == MODEL_CHECKPOINT:
            cbInstance = ModelCheckpoint(config, self._runTimeConfig)
        elif callbackId == TENSORBOARD:
            cbInstance = TensorBoard(config, self._runTimeConfig)
        else:
            errMsg = "{0} - Callback suppport for {1} not available.".format(self._log_prefix, callbackId)
            self._log.error(errMsg)
            raise ValueError(errMsg)

        if cbInstance:
            return cbInstance
