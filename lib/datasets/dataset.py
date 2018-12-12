import os, h5py, progressbar, logging
import numpy as np

class DataSet(object):
    def __init__(self):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "DATASET"
        self._hdfDataFilename = 'data.hy'
        self._textIdsFilename = 'id.txt'
        
    def prepare_h5py(self):
        lenTrainImages = len(self._trainImages)
        lenTestImages = len(self._testImages)
        lenTrainLabels = len(self._trainLabels)
        lenTestLabels = len(self._testLabels)

        train_shape = (lenTrainImages, 28, 28, 1)
        test_shape = (lenTestImages, 28, 28, 1)

        self._log.info("{0} - Preprocessing and storing dataset at: {1}".format(self._log_prefix, self._hdfDataPath))
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
            hdf5_file["train_img"][i, ...] = self._testImages[i]
           
        bar.finish()
        hdf5_file.close()
        data_ids.close()
        
        return
