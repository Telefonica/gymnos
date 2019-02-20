import os, subprocess, progressbar, logging, h5py, cv2
import dataset
import numpy as np
from keras.utils import to_categorical
from var.system_paths import *
from var.datasets import *

DEFAULT_STORAGE_IMAGE_WIDTH = 28
DEFAULT_STORAGE_IMAGE_HEIGTH = 28
DEFAULT_STORAGE_IMAGE_DEPTH = 1
MAX_TRAIN_SAMPLES = 60000
MAX_TEST_SAMPLES = 10000

class MNIST(dataset.DataSet):  
    def __init__(self, config):
        dataset.DataSet.__init__(self)
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "MNIST"
        self._sourceFiles = [ 'train-images-idx3-ubyte.gz',          # train images
                              'train-labels-idx1-ubyte.gz',
                              't10k-images-idx3-ubyte.gz',           # test images
                              't10k-labels-idx1-ubyte.gz' ]
        self._datasetUrl = 'http://yann.lecun.com/exdb/mnist/'
        self._datasetLocalDir = os.path.join(DATASETS_PATH, MNIST_DIGITS)
        self._hdfDataPath = os.path.join(self._datasetLocalDir, self._hdfDataFilename)
        self._textIdsPath = os.path.join(self._datasetLocalDir, self._textIdsFilename)
        self.dataSetId = config["id"]
        self._image_width = config["properties"]["image_width"]
        self._image_height = config["properties"]["image_height"]
        self._image_depth = config["properties"]["image_depth"]
        self._numFitSamples = config["samples"]["fit"]
        self._numValidationSamples = config["samples"]["validation"]
        self._numTestSamples = config["samples"]["test"]
        self.__checkSplitConsistency()
        
    def getSamples(self):
        return self._fitSamples, self._valSamples, self._testSamples 

    def getLabels(self):
        return self._fitLabels, self._valLabels, self._testLabels

    def load(self):
        self._data = h5py.File(self._hdfDataPath, 'r')
        self._fitSamples = self._data["train_img"][:self._numFitSamples]
        self._valSamples = self._data["train_img"][self._numFitSamples:]
        self._testSamples = self._data["test_img"][:self._numTestSamples]
        self._fitLabels = self._data["train_labels"][:self._numFitSamples]
        self._valLabels = self._data["train_labels"][self._numFitSamples:]
        self._testLabels = self._data["test_labels"][:self._numTestSamples]
        self.__preprocess()

    def download(self):
        os.mkdir(self._datasetLocalDir)
        bar = progressbar.ProgressBar( maxval=100,
                                       widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                       progressbar.Percentage()] )
        bar.start()
        i=1
        progress_rate=100/len(self._sourceFiles)
        for k in self._sourceFiles:
            url = (self._datasetUrl+k).format(**locals())
            target_path = os.path.join(self._datasetLocalDir, k)
            cmd = ['curl', url, '-o', target_path, '-s']
            subprocess.call(cmd)
            bar.update(i*progress_rate)
            cmd = ['gzip', '-d', target_path]
            subprocess.call(cmd)
            i+=1

        bar.finish()

        self.__defaultStorage()
    
    def __preprocess(self):
        # Convert 28x28 grayscale to WIDTH x HEIGHT rgb channels
        self._log.debug("{0} - __preprocess: original fitSamples shape = {1}".format(self._log_prefix, self._fitSamples.shape))
        self._log.debug("{0} - __preprocess: original fitLabels shape = {1}".format(self._log_prefix, self._fitLabels.shape))
        trainSamples = np.concatenate((self._fitSamples, self._valSamples, self._testSamples), axis = 0)
        dim = (self._image_width, self._image_height)
        
        def to_rgb(img):
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
            img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
            return img_rgb
        rgb_list = []

        # Convert trainSamples to WIDTH x HEIGHT rgb values
        for i in range(len(trainSamples)):
            rgb = to_rgb(trainSamples[i])
            rgb_list.append(rgb)
        rgb_arr = np.stack([rgb_list],axis=4)
        trainSamples = np.squeeze(rgb_arr, axis=4)
        self._fitSamples = trainSamples[:self._numFitSamples]
        self._valSamples = trainSamples[self._numFitSamples:self._numFitSamples+self._numValidationSamples]
        self._testSamples = trainSamples[self._numFitSamples+self._numValidationSamples:]

        # Convert to one-hot encoding
        self._fitLabels = to_categorical(self._fitLabels)
        self._valLabels = to_categorical(self._valLabels)
        self._testLabels = to_categorical(self._testLabels)
        self._log.debug("{0} - __preprocess: preprocessed fitSamples shape = {1}".format(self._log_prefix, self._fitSamples.shape))
        self._log.debug("{0} - __preprocess: preprocessed fitLabels shape = {1}".format(self._log_prefix, self._fitLabels.shape))
        self._log.debug("{0} - __preprocess: preprocessed valSamples shape = {1}".format(self._log_prefix, self._valSamples.shape))
        self._log.debug("{0} - __preprocess: preprocessed valLabels shape = {1}".format(self._log_prefix, self._valLabels.shape))
        self._log.debug("{0} - __preprocess: preprocessed testSamples shape = {1}".format(self._log_prefix, self._testSamples.shape))
        self._log.debug("{0} - __preprocess: preprocessed testLabels shape = {1}".format(self._log_prefix, self._testLabels.shape))
        
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

    def __defaultStorage(self):
        '''
           Standard treatment on the dataset before storage. 
           Predefined image shape by default: (width=28, height=28, depth=1).
           H5PY format will be used for performance reasons.
        '''

        fd = open(os.path.join(self._datasetLocalDir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._trainImages = loaded[16:].reshape(( MAX_TRAIN_SAMPLES,
                                                  DEFAULT_STORAGE_IMAGE_WIDTH,
                                                  DEFAULT_STORAGE_IMAGE_HEIGTH,
                                                  DEFAULT_STORAGE_IMAGE_DEPTH) ).astype(np.float)

        fd = open(os.path.join(self._datasetLocalDir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._trainLabels = np.asarray(loaded[8:].reshape((MAX_TRAIN_SAMPLES)).astype(np.float))

        fd = open(os.path.join(self._datasetLocalDir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._testImages = loaded[16:].reshape((  MAX_TEST_SAMPLES,
                                                  DEFAULT_STORAGE_IMAGE_WIDTH,
                                                  DEFAULT_STORAGE_IMAGE_HEIGTH,
                                                  DEFAULT_STORAGE_IMAGE_DEPTH) ).astype(np.float)

        fd = open(os.path.join(self._datasetLocalDir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._testLabels = np.asarray(loaded[8:].reshape((MAX_TEST_SAMPLES)).astype(np.float))

        self.__prepare_h5py()

        for k in self._sourceFiles:
            cmd = ['rm', '-f', os.path.join(self._datasetLocalDir, k[:-3])]
            subprocess.call(cmd)

    def __prepare_h5py(self):
        lenTrainImages = len(self._trainImages)
        lenTestImages = len(self._testImages)
        lenTrainLabels = len(self._trainLabels)
        lenTestLabels = len(self._testLabels)

        train_shape = ( lenTrainImages, 
                        DEFAULT_STORAGE_IMAGE_WIDTH, 
                        DEFAULT_STORAGE_IMAGE_HEIGTH, 
                        DEFAULT_STORAGE_IMAGE_DEPTH )

        test_shape = ( lenTestImages, 
                        DEFAULT_STORAGE_IMAGE_WIDTH, 
                        DEFAULT_STORAGE_IMAGE_HEIGTH, 
                        DEFAULT_STORAGE_IMAGE_DEPTH )

        self._log.info("{0} - Storing dataset at: {1}".format(self._log_prefix, self._hdfDataPath))
        bar = progressbar.ProgressBar( maxval=100,
                                       widgets=[progressbar.Bar('=', '[', ']'),
                                       ' ', 
                                       progressbar.Percentage()] )
        bar.start()

        hdf5_file = h5py.File(self._hdfDataPath, 'w')
        data_ids = open(self._textIdsPath, 'w')

        hdf5_file.create_dataset("train_labels", (lenTrainLabels,), np.uint8)
        hdf5_file.create_dataset("test_labels", (lenTestLabels,), np.uint8)
        hdf5_file["train_labels"][...] = self._trainLabels
        hdf5_file["test_labels"][...] = self._testLabels
        
        hdf5_file.create_dataset("train_img", train_shape, np.uint8)
        hdf5_file.create_dataset("test_img", test_shape, np.uint8)        
        
        for i in range(lenTrainImages):
            if i%(lenTrainImages/100)==0:
                bar.update(i/(lenTrainImages/100))
            hdf5_file["train_img"][i, ...] = self._trainImages[i]
        
        for i in range(lenTestImages):
            if i%(lenTestImages/100)==0:
                bar.update(i/(lenTestImages/100))
            hdf5_file["test_img"][i, ...] = self._testImages[i]
           
        bar.finish()
        hdf5_file.close()
        data_ids.close()
        
        return