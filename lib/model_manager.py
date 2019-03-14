from .log import logger

from .model_factory import ModelFactory


class ModelManager(object):

    def __init__(self, config):
        self._log = logger.get_logger()
        self._log_prefix = logger.setup_prefix(__class__)
        self._modelId = config["id"]
        mf = ModelFactory(config)
        self._model = mf.factory()

    def getModel(self):
        self._model.lookForModelSource()
        # self._model.lookForPretrainedWeights()
        return self._model
