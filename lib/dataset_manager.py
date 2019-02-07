
import os, logging, inspect
import keras.datasets
from var.system_paths import *
from dataset_factory import DataSetFactory

class DataSetManager(object):
    def __init__(self, config):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "DATA_SET_MGR"
        self._config = config
        self._dataSetId = config["id"]
        dsf = DataSetFactory(config)
        self._ds = dsf.factory()

    def loadDataSet(self):
       self.__lookForDataSetSource()
    
    def getSamplesForTraining(self):
        return self._ds.getSamples()

    def getLabelsForTraining(self):
        return  self._ds.getLabels()

    def __dataSetInLocalVolume(self):
        retval = False
        targetDir = '{0}/{1}'.format(DATASETS_PATH, self._dataSetId)
        if os.path.isdir(targetDir):
            self._log.info("{0} - Data set '{1}' found in local volume.".format(self._log_prefix, self._dataSetId))
            retval = True
        else:
            self._log.warning("{0} - Data set '{1}' not found in local volume.".format(self._log_prefix, self._dataSetId))
        return retval

    def __loadDataSetFromLocal(self):
        self._log.info("{0} - Loading {1} dataset ...".format(self._log_prefix, self._dataSetId))
        self._ds.load()

    def __loadDataSetFromRemote(self):
        self._log.info("{0} - Downloading {1} dataset ...".format(self._log_prefix, self._dataSetId))
        self._ds.download()
        self._log.info("{0} - Loading {1} dataset ...".format(self._log_prefix, self._dataSetId))
        self._ds.load()

    def __lookForDataSetSource(self):
        if self.__dataSetInLocalVolume():
            self.__loadDataSetFromLocal()
        else:
            self.__loadDataSetFromRemote()