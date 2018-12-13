
import os, logging, inspect
import keras.datasets
from keras.preprocessing.image import ImageDataGenerator
from var.system_paths import *
from dataset_factory import DataSetFactory

class DataSetManager(object):
    def __init__(self, dataSetId):
        self._log = logging.getLogger('aitpd')
        self._log_prefix = "DATA_SET_MGR"
        self._dataSetId = dataSetId
        dsf = DataSetFactory(self._dataSetId)
        self._ds = dsf.factory()
    
    def loadDataSet(self):
       self.__lookForDataSetSource()
    
    def getDataForTraining(self, fitSamples, valSamples=0, testSamples=0):
        return self._ds.getData(fitSamples, valSamples, testSamples)

    def getLabelsForTraining(self, fitSamples, valSamples=0, testSamples=0):
        return  self._ds.getLabels(fitSamples, valSamples, testSamples)

    def __lookForDataSetSource(self):
        if self.__dataSetInKeras():
            self.__loadDataSetFromKeras()
        else:
            if self.__dataSetInLocalVolume():
                self.__loadDataSetFromLocal()
            else:
                self.__loadDataSetFromRemote()


    def __dataSetInKeras(self):
        retval = False
        if not self._dataSetId in inspect.getmembers(keras.datasets):
            self._log.warning("{0} - Data set '{1}' not found in keras.".format(self._log_prefix, self._dataSetId))
        else:
            self._log.info("{0} - Data set '{1}' found in keras.".format(self._log_prefix, self._dataSetId))
            retval = True

        return retval


    def __dataSetInLocalVolume(self):
        retval = False
        targetDir = '{0}/{1}'.format(DATASETS_PATH, self._dataSetId)
        if os.path.isdir(targetDir):
            self._log.info("{0} - Data set '{1}' found in local volume.".format(self._log_prefix, self._dataSetId))
            retval = True
        else:
            self._log.warning("{0} - Data set '{1}' not found in local volume.".format(self._log_prefix, self._dataSetId))

        return retval


    def __loadDataSetFromKeras(self):
        self._dataSet = globals()[self._dataSetId]()
        (x_train, y_train), (x_test, y_test) = self._dataSet.load_data()
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)


    def __loadDataSetFromLocal(self):
        self._log.info("{0} - Loading {1} dataset ...".format(self._log_prefix, self._dataSetId))
        self._ds.load()
        '''
        dataSetType = self._config.dataset.type
        if dataSetType == 'image':
            self.__loadImagesFromLocalDataSet()
        elif dataSetType == 'text':
            self.__loadTextFromLocalDataSet()
        '''

    def __loadDataSetFromRemote(self):
        self._log.info("{0} - Downloading {1} dataset ...".format(self._log_prefix, self._dataSetId))
        self._ds.download()
        self._log.info("{0} - Loading {1} dataset ...".format(self._log_prefix, self._dataSetId))
        self._ds.load()


    def __loadImagesFromLocalDataSet(self):        
        train_dir = '{0}/{1}/train'.format(DATASETS_PATH, self._dataSetId)
        val_dir = '{0}/{1}/val'.format(DATASETS_PATH, self._dataSetId)
        test_dir = '{0}/{1}/test'.format(DATASETS_PATH, self._dataSetId)

        train_datagen = ImageDataGenerator(rescale=1./255)       # Pixel Normalization
        val_datagen = ImageDataGenerator(rescale=1./255)         # Pixel Normalization
        test_datagen = ImageDataGenerator(rescale=1./255)        # Pixel Normalization

        imgage_width = self._config.dataset.properties.image_width
        imgage_heigth = self._config.dataset.properties.imgage_heigth
        batch_size = self._config.dataset.properties.batch_size
        class_mode = self._config.dataset.properties.class_mode

        self._train_generator = train_datagen.flow_from_directory( train_dir, 
                                                                   target_size=(imgage_width, imgage_heigth),
                                                                   batch_size=batch_size,
                                                                   class_mode=class_mode )

        self._val_generator = val_datagen.flow_from_directory( val_dir,
                                                               target_size=(imgage_width, imgage_heigth),
                                                               batch_size=batch_size,
                                                               class_mode=class_mode )

        self._test_generator = val_datagen.flow_from_directory( test_dir,
                                                                target_size=(imgage_width, imgage_heigth),
                                                                batch_size=batch_size,
                                                                class_mode=class_mode )

    def __loadTextFromLocalDataSet(self): 
        pass