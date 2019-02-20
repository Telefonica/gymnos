import os, logging, h5py, progressbar, cv2, random
import datetime as dt
import numpy as np

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
        self._log.debug("{0} - loadRawImagesFromFolder: Loading images with:\n[\n\t - folderPath = {1}\
                                                                            \n\t - resizeParams = {2}\
                                                                            \n]".format( self._log_prefix,
                                                                                         folderPath,
                                                                                         resizeParams ) )
        imgList = []
        images = os.listdir(folderPath)
        lenImages = len(images)
        bar = progressbar.ProgressBar( maxval=100,
                                       widgets=[progressbar.Bar('=', '[', ']'),
                                       ' ', 
                                       progressbar.Percentage()] )
        bar.start()
        time_start = dt.datetime.now()
        for i,imgFileName in enumerate(images):
            if i%(lenImages/100)==0:
                bar.update(i/(lenImages/100))            
            image = cv2.imread( os.path.join(folderPath, imgFileName) )        
            image = cv2.resize( image, 
                                resizeParams, 
                                interpolation=cv2.INTER_CUBIC )
            imgList.append(image)
        imgArr = np.stack([imgList],axis=4)
        imgArr = np.squeeze(imgArr, axis=4)
        bar.finish()
        time_end = dt.datetime.now()
        self._log.debug("{0} - loadImagesFromFolder: Processed images = [{1}] in {2} seconds.".format( self._log_prefix,
                                                                                                       lenImages,
                                                                                                       (time_end-time_start).seconds) )     
        return imgArr                                                                        

    
    def prepareH5PY(self, folderPath, trainImages, trainLabels):
        h5pyFilePath = os.path.join(folderPath, self._hdfDataFilename)
        self._log.info("{0} - Preparing H5PY dataset at: {1}".format(self._log_prefix, h5pyFilePath))
        trainImagesShape = trainImages.shape
        trainLabelsShape = trainLabels.shape
        lenTrainImages = len(trainImages)
        bar = progressbar.ProgressBar( maxval=100,
                                       widgets=[progressbar.Bar('=', '[', ']'),
                                       ' ', 
                                       progressbar.Percentage()] )
        bar.start()
        hdf5_file = h5py.File(h5pyFilePath, 'w')
        hdf5_file.create_dataset("train_labels", trainLabelsShape, np.uint8)
        hdf5_file["train_labels"][...] = trainLabels
        hdf5_file.create_dataset("train_samples", trainImagesShape, np.uint8)
        for i in range(lenTrainImages):
            if i%(lenTrainImages/100)==0:
                bar.update(i/(lenTrainImages/100))
            hdf5_file["train_samples"][i, ...] = trainImages[i]
        bar.finish()
        hdf5_file.close()
        return