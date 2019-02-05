import os, logging, subprocess, zipfile
import dataset

from kaggle import api
from var.system_paths import *
from var.datasets import *

from kaggle_factory import KaggleFactory

class KaggleBase(dataset.DataSet):  
    '''
        Works as a base class for kaggle data sets 
    '''
    def __init__(self, config):
        dataset.DataSet.__init__(self)
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "KAGGLE"
        self._config = config
        self._kaggleSource = config["properties"]["kaggle_source"]
        kf = KaggleFactory(config)
        self._datasetLocalDir = os.path.join(DATASETS_PATH, KAGGLE, self._kaggleSource["id"])
        self._kf = kf.factory()
        
        
    def download(self):
        os.makedirs(self._datasetLocalDir)
        cmd = ['kaggle',  self._kaggleSource["type"], 'download', '-c', self._kaggleSource["id"], '-p', self._datasetLocalDir]
        subprocess.call(cmd)
        self.__defaultStorage()

    def getSamples(self):
        return self._kf.getSamples() 

    def getLabels(self):
        return self._kf.getLables()

    def load(self):
        self._data = h5py.File(self._hdfDataPath, 'r')
        self._fitSamples = self._data["train_img"][:self._numFitSamples]
        self._valSamples = self._data["train_img"][self._numFitSamples:]
        self._testSamples = self._data["test_img"][:self._numTestSamples]
        self._fitLabels = self._data["train_labels"][:self._numFitSamples]
        self._valLabels = self._data["train_labels"][self._numFitSamples:]
        self._testLabels = self._data["test_labels"][:self._numTestSamples]
        self.__preprocess()

    def __preprocess(self):
        self._kf.preprocess()

    def __uncompressDataSetFiles(self):
        extension = ".zip"
        for item in os.listdir(self._datasetLocalDir):
            if item.endswith(extension): 
                file_name = os.path.join(self._datasetLocalDir, item) 
                self._log.info("{0} - Unzipping {1} ...".format(self._log_prefix, file_name))
                zip_ref = zipfile.ZipFile(file_name) 
                zip_ref.extractall(self._datasetLocalDir) 
                zip_ref.close() 
                os.remove(file_name) 

    def __defaultStorage(self):
        '''
           Standard treatment on the dataset before storage. 
        '''
        self.__uncompressDataSetFiles()
        self._kf.defaultStorage(self._datasetLocalDir)
