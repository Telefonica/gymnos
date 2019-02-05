import os, logging, subprocess, zipfile

MAX_TRAIN_SAMPLES = 25000
MAX_TEST_SAMPLES = 12500

class DogsVsCats(object):  
    def __init__(self, config):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "KAGGLE_DOGS_VS_CATS"
        self._config = config
        self._numFitSamples = config["samples"]["fit"]
        self._numValidationSamples = config["samples"]["validation"]
        self._numTestSamples = config["samples"]["test"]
        self.__checkSplitConsistency()

    def defaultStorage(self, localDir):
        self._localDir = localDir
        self._log.debug("{0} - defaultStorage: Starting at = {1}".format(self._log_prefix, self._localDir))
        fitDir = os.path.join(self._localDir, "fit")
        valDir = os.path.join(self._localDir, "val")
        
        os.makedirs(os.path.join(fitDir, "dogs"))
        os.makedirs(os.path.join(fitDir, "cats"))
        os.makedirs(os.path.join(valDir, "dogs"))
        os.makedirs(os.path.join(valDir, "cats"))

        #TODO: move smaples from trainDir to subfolder structure


    def getSamples(self):
        return self._fitSamples, self._valSamples, self._testSamples 

    def getLabels(self):
        return self._fitLabels, self._valLabels, self._testLabels
    
    def preprocess(self):
        self._log.info("{0} - Image preprocessing started.".format(self._log_prefix))
        train_dir = 'kaggleDataSets/dogsVsCats2013/train'
        val_dir = 'kaggleDataSets/dogsVsCats2013/validation'

        self.__dataAugmentation()
        self.__getSamplesFromGenerator()

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

    def __dataAugmentation(self):
        # note: 'batch_size = 1' since samples will be returned as a whole array
        self._log.debug("{0} - __dataAugmentation: Applying Keras ImageDataGenerator to = {1}".format(self._log_prefix, self._localDir))
        from keras.preprocessing.image import ImageDataGenerator

        # augmentation config for fit set
        fitDatagen = ImageDataGenerator( rescale=1./255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True )
        self._fitGenerator = fitDatagen.flow_from_directory( train_dir, 
                                                             target_size=(150, 150),
                                                             batch_size=1,
                                                             class_mode='binary' )
        self._log.debug("{0} - __dataAugmentation: fitGenerator samples shape = {1}".format(self._log_prefix, self._fitGenerator[0][0].shape))
        self._log.debug("{0} - __dataAugmentation: fitGenerator labels shape = {1}".format(self._log_prefix, self._fitGenerator[0][1].shape))

        # augmentation config for validation set                                  
        valDatagen = ImageDataGenerator( rescale=1./255,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True )
        self._valGenerator = valDatagen.flow_from_directory( val_dir,
                                                             target_size=(150, 150),
                                                             batch_size=1,
                                                             class_mode='binary' )
        self._log.debug("{0} - __dataAugmentation: valGenerator samples shape = {1}".format(self._log_prefix, self._valGenerator[0][0].shape))
        self._log.debug("{0} - __dataAugmentation: valGenerator labels shape = {1}".format(self._log_prefix, self._valGenerator[0][1].shape))

        # augmentation config for validation set                                  
        testDatagen = ImageDataGenerator( rescale=1./255 )
        self._testGenerator = testDatagen.flow_from_directory( test_dir,
                                                              target_size=(150, 150),
                                                              batch_size=1,
                                                              class_mode='binary' )
        self._log.debug("{0} - __dataAugmentation: testGenerator samples shape = {1}".format(self._log_prefix, self._testGenerator[0][0].shape))
        self._log.debug("{0} - __dataAugmentation: testGenerator labels shape = {1}".format(self._log_prefix, self._testGenerator[0][1].shape))
        
    def __getSamplesFromGenerator(self):
        batch_index = 0
        while batch_index <= self._fitGenerator.batch_index:
            data = train_generator.next()
            self._fitSamples.append(data[0])
            self._fitLables.append(data[1])
            batch_index += 1
        
        batch_index = 0
        while batch_index <= self._valGenerator.batch_index:
            data = train_generator.next()
            self._valSamples.append(data[0])
            self._valLables.append(data[1])
            batch_index += 1

        batch_index = 0
        while batch_index <= self._testGenerator.batch_index:
            data = test_generator.next()
            self._testSamples.append(data[0])
            self._testLables.append(data[1])
            batch_index += 1

   