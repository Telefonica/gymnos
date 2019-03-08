import os
import logging
import h5py
import cv2
import time
import numpy as np

from tqdm import tqdm


class DataSet(object):
    def __init__(self):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "DATASET"
        self._hdfDataFilename = 'training-data-set.hdf5'
        self._textIdsFilename = 'id.txt'

    def loadRawImagesFromFolder(self, folderPath, resizeParams):
        '''
             Load set of raw images from folder
             Notes:
              - resize required as images comes with different sizes
              - downsampling interpolation applied to optimize performance
        '''
        self._log.debug("{0} - loadRawImagesFromFolder: Loading images with:\n[\n\t - folderPath = {1}"
                        "\n\t - resizeParams = {2}"
                        "\n]".format(self._log_prefix,
                                     folderPath,
                                     resizeParams))
        imgList = []
        images = os.listdir(folderPath)

        time_start = time.time()
        for i, imgFileName in enumerate(tqdm(images)):
            image = cv2.imread(os.path.join(folderPath, imgFileName))
            image = cv2.resize(image,
                               resizeParams,
                               interpolation=cv2.INTER_CUBIC)
            imgList.append(image)

        imgArr = np.stack([imgList], axis=4)
        imgArr = np.squeeze(imgArr, axis=4)

        self._log.debug("{0} - loadImagesFromFolder: Processed images = [{1}] in {2:.2f} seconds.".format(
                        self._log_prefix, len(images), (time.time() - time_start)))
        return imgArr


    def prepareH5PY(self, folderPath, trainImages, trainLabels):
        h5pyFilePath = os.path.join(folderPath, self._hdfDataFilename)
        self._log.info("{0} - Preparing H5PY dataset at: {1}".format(self._log_prefix, h5pyFilePath))

        with h5py.File(h5pyFilePath, 'w') as hdf5_file:
            hdf5_file.create_dataset("train_labels", data=trainLabels, dtype=np.uint8)
            hdf5_file.create_dataset("train_samples", data=trainImages, dtype=np.uint8)
