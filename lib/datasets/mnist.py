import os, tarfile, subprocess, progressbar
import dataset
import numpy as np
from var.system_paths import *
from var.datasets import *

class MNIST(dataset.DataSet):
    FILES = [ 'train-images-idx3-ubyte.gz',          # train images
              'train-labels-idx1-ubyte.gz',
              't10k-images-idx3-ubyte.gz',           # test images
              't10k-labels-idx1-ubyte.gz']

    def __init__(self):
        self._datasetLocalDir = os.path.join(DATASETS_PATH, MNIST_DIGITS)
        self._datasetUrl = 'http://yann.lecun.com/exdb/mnist/'
        os.mkdir(self._datasetLocalDir)
        
    def download(self):
        bar = progressbar.ProgressBar( maxval=100,
                                        widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                        progressbar.Percentage()] )
        bar.start()
        i=1
        progress_rate=100/len(self.FILES)
        for k in self.FILES:
            url = (self._datasetUrl+k).format(**locals())
            target_path = os.path.join(self._datasetLocalDir, k)
            cmd = ['curl', url, '-o', target_path, '-s']
            #print('Downloading ', k)
            #print "Executing [{0}] command.".format(cmd)
            subprocess.call(cmd)
            bar.update(i*progress_rate)
            cmd = ['gzip', '-d', target_path]
            #print('Unzip ', k)
            subprocess.call(cmd)
            i+=1

        bar.finish()

    def load(self):
        num_mnist_train = 60000
        num_mnist_test = 10000

        fd = open(os.path.join(self._datasetLocalDir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._trainImages = loaded[16:].reshape((num_mnist_train,28,28,1)).astype(np.float)

        fd = open(os.path.join(self._datasetLocalDir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._trainLabels = np.asarray(loaded[8:].reshape((num_mnist_train)).astype(np.float))

        fd = open(os.path.join(self._datasetLocalDir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._testImages = loaded[16:].reshape((num_mnist_test,28,28,1)).astype(np.float)

        fd = open(os.path.join(self._datasetLocalDir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        self._testLabels = np.asarray(loaded[8:].reshape((num_mnist_test)).astype(np.float))

        self.prepare_h5py()
        for k in self.FILES:
            cmd = ['rm', '-f', os.path.join(self._datasetLocalDir, k[:-3])]
            subprocess.call(cmd)