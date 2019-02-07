import os, logging, subprocess, zipfile, random, h5py
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
        #self._fitDir = os.path.join(self._datasetLocalDir, "fit")
        #self._valDir = os.path.join(self._datasetLocalDir, "val")
        #self._testDir = os.path.join(self._datasetLocalDir, "test")
        self._numFitSamples = config["samples"]["fit"]
        self._numValidationSamples = config["samples"]["validation"]
        self._numTestSamples = config["samples"]["test"]
        self._kaggleService = config["properties"]["service"]
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
        self._fitSamples = self._data["train_img"][:self._numFitSamples]
        self._valSamples = self._data["train_img"][self._numFitSamples:]
        self._testSamples = self._data["train_img"][:self._numTestSamples]
        self._fitLabels = self._data["train_labels"][:self._numFitSamples]
        self._valLabels = self._data["train_labels"][self._numFitSamples:]
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
        #self.__dataAugmentation()
        #self.__getSamplesFromGenerator()
   
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
        self._testImages = self.loadRawImagesFromFolder( self._testDefaultDir, (DEFAULT_STORAGE_IMAGE_WIDTH, DEFAULT_STORAGE_IMAGE_HEIGTH) )
        self._trainLabels = self.__loadLabelsFromFolder( self._trainDefaultDir)
        self.prepareH5PY(self._datasetLocalDir, self._trainImages, self._trainLabels)
        self.__clean()

    def __reduceDataSetSize(self):
        self._log.debug("{0} - __reduceDataSetSize: Reducing dataset size...".format(self._log_prefix))
        for filename in glob(os.path.join(self._datasetLocalDir, "train/*.*[0-5]*.jpg")):
            os.remove(os.path.join(self._datasetLocalDir, filename))
        for filename in glob(os.path.join(self._datasetLocalDir, "test1/*[0-5]*.jpg")):
            os.remove(os.path.join(self._datasetLocalDir, filename))


    '''
    def __dataAugmentation(self):
        # note: 'batch_size = 1' since samples will be returned as a whole array
        self._log.debug("{0} - __dataAugmentation: Applying Keras ImageDataGenerator to = {1}".format(self._log_prefix, self._datasetLocalDir))
        from keras.preprocessing.image import ImageDataGenerator

        # augmentation config for fit set
        fitDatagen = ImageDataGenerator( rescale=1./255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True )
        self._fitGenerator = fitDatagen.flow_from_directory( self._fitDir, 
                                                             target_size=(150, 150),
                                                             batch_size=1,
                                                             color_mode="rgb",
                                                             class_mode='binary',
                                                             shuffle=True,
                                                             seed=42 )
        self._log.debug("{0} - __dataAugmentation: fitGenerator samples shape = {1}".format(self._log_prefix, self._fitGenerator[0][0].shape))
        self._log.debug("{0} - __dataAugmentation: fitGenerator labels shape = {1}".format(self._log_prefix, self._fitGenerator[0][1].shape))

        # augmentation config for validation set                                  
        valDatagen = ImageDataGenerator( rescale=1./255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True )
        self._valGenerator = valDatagen.flow_from_directory( self._valDir,
                                                             target_size=(150, 150),
                                                             batch_size=1,
                                                             color_mode="rgb",
                                                             class_mode='binary',
                                                             shuffle=True,
                                                             seed=42 )
        self._log.debug("{0} - __dataAugmentation: valGenerator samples shape = {1}".format(self._log_prefix, self._valGenerator[0][0].shape))
        self._log.debug("{0} - __dataAugmentation: valGenerator labels shape = {1}".format(self._log_prefix, self._valGenerator[0][1].shape))

        # augmentation config for validation set                                  
        testDatagen = ImageDataGenerator( rescale=1./255 )
        self._testGenerator = testDatagen.flow_from_directory( self._testDir,
                                                               target_size=(150, 150),
                                                               batch_size=1,
                                                               color_mode="rgb",
                                                               class_mode=None,
                                                               shuffle=False,
                                                               seed=42 )
        self._log.debug("{0} - __dataAugmentation: testGenerator samples shape = {1}".format(self._log_prefix, self._testGenerator))      

    def __defaultStorage(self):
        self._log.debug("{0} - defaultStorage: Starting at = {1}".format(self._log_prefix, self._datasetLocalDir))
        fitDirDogs = os.path.join(self._fitDir, "dogs")
        fitDirCats = os.path.join(self._fitDir, "cats")
        valDirDogs = os.path.join(self._valDir, "dogs")
        valDirCats = os.path.join(self._valDir, "cats")
        os.makedirs(fitDirDogs)
        os.makedirs(fitDirCats)
        os.makedirs(valDirDogs)
        os.makedirs(valDirCats)
        os.rename(os.path.join(self._datasetLocalDir, DEFAULT_TEST_FOLDER_NAME), self._testDir)
        # Move samples from trainDir to subfolder structure
        fromPathBase = os.path.join(self._datasetLocalDir, DEFAULT_TRAIN_FOLDER_NAME)
        files = os.listdir(fromPathBase)
        random.shuffle(files)
        for index, f in enumerate(files):
            toPath = ""
            if index < self._numFitSamples:
                toPath = os.path.join(fitDirDogs, f) if f.split(".")[0] == "dog" else os.path.join(fitDirCats, f)
            else:
                toPath = os.path.join(valDirDogs, f) if f.split(".")[0] == "dog" else os.path.join(valDirCats, f)
            fromPath = os.path.join(fromPathBase, f)
            os.rename(fromPath, toPath)

    def __getSamplesFromGenerator(self):
        self._fitSamples = []
        self._valSamples = []
        self._testSamples = []
        self._fitLabels = []
        self._valLabels = []
        self._testLabels = []

        batch_index = 0
        while batch_index <= self._fitGenerator.batch_index:
            data = self._fitGenerator.next()
            print "fit: batch: ", batch_index, "generator index: ", self._fitGenerator.batch_index
            self._fitSamples.append(data[0])
            self._fitLabels.append(data[1])
            #print "_fitSamples: ", self._fitSamples, "_fitLabels ", self._fitLabels
            batch_index += 1
        
        batch_index = 0
        while batch_index <= self._valGenerator.batch_index:
            data = self._valGenerator.next()
            print "val: batch: ", batch_index, "generator index: ", self._valGenerator.batch_index
            self._valSamples.append(data[0])
            self._valLabels.append(data[1])
            batch_index += 1
        
        batch_index = 0
        while batch_index <= self._testGenerator.batch_index:
            data = self._testGenerator.next()
            print "test: batch: ", batch_index, "generator index: ", self._testGenerator.batch_index
            self._testSamples.append(data[0])
            self._testLabels.append(data[1])
            batch_index += 1
    '''