import os, tarfile, subprocess, progressbar, logging, h5py
import dataset
import numpy as np
from var.system_paths import *
from var.datasets import *

class MNIST(dataset.DataSet):  
    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self._ids)

    def __init__(self):
        dataset.DataSet.__init__(self)
        self.dataSetId = MNIST_DIGITS
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
        self._maxTrainSamples = 60000
        self._maxTestSamples = 10000
        
    def getData(self, numFitSamples, numValSamples, numTestSamples):
        self.__checkSplitConsistency(numFitSamples, numValSamples, numTestSamples)
        fitSamples = self._data["train_img"][:numFitSamples]
        valSamples = self._data["train_img"][numFitSamples:]
        testSamples = self._data["test_img"][:numTestSamples]
        return fitSamples, valSamples, testSamples 

    def getLabels(self, numFitSamples, numValSamples, numTestSamples):
        fitLabels = self._data["train_labels"][:numFitSamples]
        valLabels = self._data["train_labels"][numFitSamples:]
        testLabels = self._data["test_labels"][:numTestSamples]
        return fitLabels, valLabels, testLabels

    def load(self):
        self._data = h5py.File(self._hdfDataPath, 'r')
        with open(self._textIdsPath, 'r') as fp:
            self._ids = [s.strip() for s in fp.readlines() if s]

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

        self.__preprocessDataSet()
    
    def __checkSplitConsistency(self, fitSamples, valSamples, testSamples):
        errMsg = None
        totalTrain = fitSamples + valSamples
        if testSamples > self._maxTestSamples:
            errMsg = "{0} - Number of test samples out of range: {1} .".format(self._log_prefix, testSamples)
        if totalTrain > self._maxTrainSamples:
            errMsg = "{0} - Number of train samples out of range: {1} .".format(self._log_prefix, totalSamples)
        if errMsg is not None:
            self._log.error(errMsg)
            raise ValueError(errMsg)

    def __preprocessDataSet(self):
        '''
           Standard treatment on the dataset before storage
        '''
        fd = open(os.path.join(self._datasetLocalDir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._trainImages = loaded[16:].reshape((self._maxTrainSamples,28,28,1)).astype(np.float)

        fd = open(os.path.join(self._datasetLocalDir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._trainLabels = np.asarray(loaded[8:].reshape((self._maxTrainSamples)).astype(np.float))

        fd = open(os.path.join(self._datasetLocalDir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._testImages = loaded[16:].reshape((self._maxTestSamples,28,28,1)).astype(np.float)

        fd = open(os.path.join(self._datasetLocalDir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._testLabels = np.asarray(loaded[8:].reshape((self._maxTestSamples)).astype(np.float))

        self.prepare_h5py()

        for k in self._sourceFiles:
            cmd = ['rm', '-f', os.path.join(self._datasetLocalDir, k[:-3])]
            subprocess.call(cmd)
