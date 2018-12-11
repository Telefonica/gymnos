import os, tarfile, subprocess, argparse, h5py, json, logging, progressbar
import numpy as np
from datasets.mnist import *
from var.datasets import *

class DataSetFactory(object):
    def __init__(self, dataSetId):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "DATA_SET_FACTORY"
        self._dataSetId = dataSetId

    def factory(self):
        if self._dataSetId == MNIST_DIGITS: 
            self._log.debug("{0} - Instantiating {1} dataset ...".format(self._log_prefix, MNIST_DIGITS))
            return MNIST()
        '''
        if type == "CIFAR10": return CIFAR10()
        if type == "ImageNet": return ImageNet()
        if type == "IMDBDataBase": return IMDBDataBase()
        if type == "Kaggle": return Kaggle()
        '''

        errMsg = "{0} - Data set suppport for {1} not available.".format(self._log_prefix, self._dataSetId)
        self._log.error(errMsg)
        raise ValueError(errMsg)

'''


    def download_svhn(download_path):
        data_dir = os.path.join(download_path, 'svhn')

        import scipy.io as sio
        # svhn file loader
        def svhn_loader(url, path):
            cmd = ['curl', url, '-o', path]
            subprocess.call(cmd)
            m = sio.loadmat(path)
            return m['X'], m['y']

        if check_file(data_dir):
            print('SVHN was downloaded.')
            return

        data_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
        train_image, train_label = svhn_loader(data_url, os.path.join(data_dir, 'train_32x32.mat'))

        data_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
        test_image, test_label = svhn_loader(data_url, os.path.join(data_dir, 'test_32x32.mat'))

        prepare_h5py(np.transpose(train_image, (3, 0, 1, 2)), train_label,
                    np.transpose(test_image, (3, 0, 1, 2)), test_label, data_dir)

        cmd = ['rm', '-f', os.path.join(data_dir, '*.mat')]
        subprocess.call(cmd)

    def download_cifar10(download_path):
        data_dir = os.path.join(download_path, 'cifar10')

        # cifar file loader
        def unpickle(file):
            import cPickle
            with open(file, 'rb') as fo:
                dict = cPickle.load(fo)
            return dict

        if check_file(data_dir):
            print('CIFAR was downloaded.')
            return

        data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        k = 'cifar-10-python.tar.gz'
        target_path = os.path.join(data_dir, k)
        print(target_path)
        cmd = ['curl', data_url, '-o', target_path]
        print('Downloading CIFAR10')
        subprocess.call(cmd)
        tarfile.open(target_path, 'r:gz').extractall(data_dir)

        num_cifar_train = 50000
        num_cifar_test = 10000

        target_path = os.path.join(data_dir, 'cifar-10-batches-py')
        train_image = []
        train_label = []
        for i in range(5):
            fd = os.path.join(target_path, 'data_batch_'+str(i+1))
            dict = unpickle(fd)
            train_image.append(dict['data'])
            train_label.append(dict['labels'])

        train_image = np.reshape(np.stack(train_image, axis=0), [num_cifar_train, 32*32*3])
        train_label = np.reshape(np.array(np.stack(train_label, axis=0)), [num_cifar_train])

        fd = os.path.join(target_path, 'test_batch')
        dict = unpickle(fd)
        test_image = np.reshape(dict['data'], [num_cifar_test, 32*32*3])
        test_label = np.reshape(dict['labels'], [num_cifar_test])

        prepare_h5py(train_image, train_label, test_image, test_label, data_dir, [32, 32, 3])

        cmd = ['rm', '-f', os.path.join(data_dir, 'cifar-10-python.tar.gz')]
        subprocess.call(cmd)
        cmd = ['rm', '-rf', os.path.join(data_dir, 'cifar-10-batches-py')]
        subprocess.call(cmd)
'''
