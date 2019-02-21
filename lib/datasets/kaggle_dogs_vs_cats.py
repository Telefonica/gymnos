import os, logging, subprocess, random, h5py
import numpy as np
import dataset
from glob import glob
from var.system_paths import *
from var.datasets import *
from services.kaggle_base import KaggleBase

DEFAULT_STORAGE_IMAGE_WIDTH = 150
DEFAULT_STORAGE_IMAGE_HEIGTH = 150
DEFAULT_STORAGE_IMAGE_DEPTH = 3
DEFAULT_TRAIN_FOLDER_NAME = "train"
DEFAULT_TEST_FOLDER_NAME = "test1"
DEFAULT_LABEL_CAT = 0
DEFAULT_LABEL_DOG = 1
MAX_TRAIN_SAMPLES = 25000
MAX_TEST_SAMPLES = 12500

class KaggleDogsVsCats(dataset.DataSet):  
    def __init__(self, config):
        dataset.DataSet.__init__(self)
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "KAGGLE_DOGS_VS_CATS"
        self._config = config
        self._datasetLocalDir = os.path.join(DATASETS_PATH, config["id"])
        self._trainDefaultDir = os.path.join(self._datasetLocalDir, DEFAULT_TRAIN_FOLDER_NAME)
        self._testDefaultDir = os.path.join(self._datasetLocalDir, DEFAULT_TEST_FOLDER_NAME)
        self._hdfDataPath = os.path.join(self._datasetLocalDir, self._hdfDataFilename)
        self._numFitSamples = config["samples"]["fit"]
        self._numValidationSamples = config["samples"]["validation"]
        self._numTestSamples = config["samples"]["test"]
        self._kaggleService = config["properties"]["service"]
        self._applyShuffle = config["properties"]["shuffle"]
        self.__checkSplitConsistency()
    
    def getSamples(self):
        return self._fitSamples, self._valSamples, self._testSamples 

    def getLabels(self):
        return self._fitLabels, self._valLabels, self._testLabels

    def download(self):
        KaggleBase().download( self._datasetLocalDir,
                               self._kaggleService["type"],
                               self._kaggleService["id"] )
        #self.__reduceDataSetSize()
        self.__defaultStorage()

    def load(self):
        self._data = h5py.File(self._hdfDataPath, 'r')
        self._fitSamples = self._data["train_samples"][:self._numFitSamples]
        self._valSamples = self._data["train_samples"][self._numFitSamples:(self._numFitSamples+self._numValidationSamples)]
        self._testSamples = self._data["train_samples"][:self._numTestSamples]
        self._fitLabels = self._data["train_labels"][:self._numFitSamples]
        self._valLabels = self._data["train_labels"][self._numFitSamples:(self._numFitSamples+self._numValidationSamples)]
        self._testLabels = self._data["train_labels"][:self._numTestSamples]       
        self.__preprocess()

    def __checkSplitConsistency(self):
        errMsg = None
        numTrainSamples = self._numFitSamples + self._numValidationSamples
        if self._numTestSamples > MAX_TEST_SAMPLES:
            errMsg = "{0} - Number of test samples out of range: {1} .".format(self._log_prefix, self._numTestSamples)
        if numTrainSamples > MAX_TRAIN_SAMPLES:
            errMsg = "{0} - Number of train samples out of range: {1} .".format(self._log_prefix, numTrainSamples)
        if errMsg is not None:
            self._log.error(errMsg)
            raise ValueError(errMsg)
    
    def __preprocess(self):
        self._log.info("{0} - Image preprocessing started.".format(self._log_prefix))
   
    def __loadLabelsFromFolder(self, folderPath):
        labelList = []
        for i,imgFileName in enumerate(os.listdir(folderPath)):    
            label = DEFAULT_LABEL_DOG if imgFileName.split(".")[0] == "dog" else DEFAULT_LABEL_CAT
            labelList.append(label)
            labelArr = np.stack([labelList], axis=1)
            labelArr = np.squeeze(labelArr, axis=1)
        return labelArr
        
    def __clean(self):
        self._log.debug("{0} - __clean: Cleaning original files...".format(self._log_prefix))
        cmd = ['rm', '-rf', os.path.join(self._datasetLocalDir, DEFAULT_TRAIN_FOLDER_NAME)]
        subprocess.call(cmd)
        cmd = ['rm', '-rf', os.path.join(self._datasetLocalDir, DEFAULT_TEST_FOLDER_NAME)]
        subprocess.call(cmd)
        cmd = ['rm', '-rf', os.path.join(self._datasetLocalDir, "sampleSubmission.csv")]
        subprocess.call(cmd)

    def __defaultStorage(self):
        '''
           Standard treatment on the dataset before storage. 
           H5PY format will be used for performance reasons.
        '''
        self._trainImages = self.loadRawImagesFromFolder( self._trainDefaultDir, (DEFAULT_STORAGE_IMAGE_WIDTH, DEFAULT_STORAGE_IMAGE_HEIGTH) )
        self._trainLabels = self.__loadLabelsFromFolder( self._trainDefaultDir )
        self.__shuffleDataset() if self._applyShuffle is True else None
        self.prepareH5PY(self._datasetLocalDir, self._trainImages, self._trainLabels)
        self.__clean()

    def __reduceDataSetSize(self):
        self._log.debug("{0} - __reduceDataSetSize: Reducing dataset size...".format(self._log_prefix))
        for filename in glob(os.path.join(self._datasetLocalDir, "train/*.*[0-5]*.jpg")):
            os.remove(os.path.join(self._datasetLocalDir, filename))
        for filename in glob(os.path.join(self._datasetLocalDir, "test1/*[0-5]*.jpg")):
            os.remove(os.path.join(self._datasetLocalDir, filename))

    def __shuffleDataset(self):
        shuffledList = []
        for i, image in enumerate(self._trainImages):
            shuffledList.append([self._trainImages[i], self._trainLabels[i]])
        np.random.shuffle(shuffledList)
        for i, item in enumerate(shuffledList):
            self._trainImages[i] = item[0]
            self._trainLabels[i] = item[1]
