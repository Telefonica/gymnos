import os, logging, inspect
import keras.applications
from var.system_paths import *
from var.models import *
from model_factory import ModelFactory

class ModelManager(object):
    def __init__(self, modelId):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "MODEL_MGR"
        self._modelId = modelId
        mf = ModelFactory(self._modelId)
        self._model = mf.factory()
        
    def getModel(self): 
        self._model.lookForModelSource()
        self._model.lookForPretrainedWeights()
        return self._model
