import os, tarfile, subprocess, argparse, h5py, json, logging, progressbar
import numpy as np

BASE_PATH = '/home/sysadmin/aitp/'
SYS_CONFIG_PATH = BASE_PATH + 'config/system.json'

with open(SYS_CONFIG_PATH, 'rb') as fp:
  sys_config = json.load(fp)

DATASETS_PATH = sys_config['paths']['datasets']

parser = argparse.ArgumentParser(description='Download dataset for SSGAN.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+', choices=['MNIST', 'SVHN', 'CIFAR10'])

MNIST_DIGITS = "mnist-digits"

class DataSetFactory(object):

    def __init__(self, dataSetId):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "DATA_SET_FACTORY"
        self._dataSetId = dataSetId

    
    def factory(self):
        class DataSet(object):
            def __init__(self):
                pass

        class MNIST(DataSet):
            def __init__(self):
                self._datasetLocalDir = os.path.join(DATASETS_PATH, MNIST_DIGITS)
                self._datasetUrl = 'http://yann.lecun.com/exdb/mnist/'
                os.mkdir(self._datasetLocalDir)
                

            def download(self):
                bar = progressbar.ProgressBar( maxval=100,
                                               widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()] )
                bar.start()
                
                keys = ['train-images-idx3-ubyte.gz',          # train images
                        'train-labels-idx1-ubyte.gz',
                        't10k-images-idx3-ubyte.gz',           # test images
                        't10k-labels-idx1-ubyte.gz']
                i=1
                progress_rate=100/len(keys)
                for k in keys:
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
        '''
        class CIFAR10(DataSet):
            pass

        class ImageNet(DataSet):
            pass

        class IMDBDataBase(DataSet):
            pass

        class Kaggle(DataSet):
            pass
        '''
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
        def prepare_h5py(train_image, train_label, test_image, test_label, data_dir, shape=None):

        image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)
        label = np.concatenate((train_label, test_label), axis=0).astype(np.uint8)

        print('Preprocessing data...')

        import progressbar
        bar = progressbar.ProgressBar(maxval=100,
                                    widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                            progressbar.Percentage()])
        bar.start()

        f = h5py.File(os.path.join(data_dir, 'data.hy'), 'w')
        data_id = open(os.path.join(data_dir,'id.txt'), 'w')
        for i in range(image.shape[0]):

            if i%(image.shape[0]/100)==0:
                bar.update(i/(image.shape[0]/100))

            grp = f.create_group(str(i))
            data_id.write(str(i)+'\n')
            if shape:
                grp['image'] = np.reshape(image[i], shape, order='F')
            else:
                grp['image'] = image[i]
            label_vec = np.zeros(10)
            label_vec[label[i]%10] = 1
            grp['label'] = label_vec.astype(np.bool)
        bar.finish()
        f.close()
        data_id.close()
        return

    def check_file(data_dir):
        if os.path.exists(data_dir):
            if os.path.isfile(os.path.join('data.hy')) and \
                os.path.isfile(os.path.join('id.txt')):
                return True
        else:
            os.mkdir(data_dir)
        return False


        num_mnist_train = 60000
        num_mnist_test = 10000

        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        train_image = loaded[16:].reshape((num_mnist_train,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        train_label = np.asarray(loaded[8:].reshape((num_mnist_train)).astype(np.float))

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        test_image = loaded[16:].reshape((num_mnist_test,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        test_label = np.asarray(loaded[8:].reshape((num_mnist_test)).astype(np.float))

        prepare_h5py(train_image, train_label, test_image, test_label, data_dir)

        for k in keys:
            cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
            subprocess.call(cmd)

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
