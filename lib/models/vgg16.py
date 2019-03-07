import os
import logging
import numpy as np

from . import model

from keras.applications.vgg16 import VGG16

class VGG16(model.Model):  
    def __init__(self, config):
        model.Model.__init__(self, config)
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "VGG16"
        self._modelInstance = None
        self._config = config
        self._modelId = config["id"]
        self._optimizer = config["options"]["compilation"]["optimizer"]
        self._loss = config["options"]["compilation"]["loss"]
        self._metrics = config["options"]["compilation"]["metrics"]
        if self._fineTunning is True:
            self._input_height = config["options"]["custom"]["fine-tunning"]["input_height"]
            self._input_width = config["options"]["custom"]["fine-tunning"]["input_width"]
            self._input_depth = config["options"]["custom"]["fine-tunning"]["input_depth"]
            self._batch_size = config["options"]["custom"]["fine-tunning"]["batch_size"]

    def compile(self):
        #TODO: Check all parameters are included, otherwise launch assert or protect code
        if self._modelInstance: 
            self._log.info("{0} - Compiling model with:\n[\n\t - loss = {1}\
                                                    \n\t - optimizer = {2}\
                                                    \n\t - metrics = {3}\
                                                    \n]".format( self._log_prefix,
                                                                 self._loss, 
                                                                 self._optimizer,
                                                                 self._metrics ))
            self._modelInstance.compile( loss=self._loss,
                                         optimizer=self._optimizer,
                                         metrics=self._metrics )
        else:
            errMsg = "{0} - Model can not be compiled. Model instance does not exist".format(self._log_prefix)
            self._log.error(errMsg)
            raise ValueError(errMsg)    

    def evaluate(self, samples, labels):
        return self._modelInstance.evaluate(samples, labels)

    def init(self):
        if self._kerasModel is True: self.__tryToLoadModelFromKeras()
        if self._localModel is True: self.__tryToLoadModelFromLocalVolume()

    def fineTune(self, fitSamples, valSamples, testSamples, fitLabels, valLabels, testLabels, trainDir):
        fineTunningFolder = "{}/fine-tunning".format(os.path.dirname(trainDir))
        if self.__lookForPreviousExtractedFeatures(fineTunningFolder) is True:
            self._log.debug("{0} - fineTune - Found previous extracted features. Loading files from {1}".format(self._log_prefix, fineTunningFolder))
            fitFeatures = np.load("{0}/fitFeatures.npz".format(fineTunningFolder))["samples"]
            valFeatures = np.load("{0}/valFeatures.npz".format(fineTunningFolder))["samples"]
            testFeatures = np.load("{0}/testFeatures.npz".format(fineTunningFolder))["samples"]
        else:  # could not found existing previous extracted features
            # Extract features
            self._log.debug("{0} - fineTune - Initiating feature extraction from conv_base ...".format(self._log_prefix))
            fitFeatures = self._conv_base.predict(np.array(fitSamples), batch_size=self._batch_size, verbose=1)
            valFeatures = self._conv_base.predict(np.array(valSamples), batch_size=self._batch_size, verbose=1)
            testFeatures = self._conv_base.predict(np.array(testSamples), batch_size=self._batch_size, verbose=1)
            # Save the features so that they can be used for future
            os.makedirs(fineTunningFolder)
            self._log.debug("{0} - fineTune - Directory created at - {1}".format(self._log_prefix, fineTunningFolder))
            self._log.debug("{0} - fineTune - Saving features at: {1}".format(self._log_prefix, fineTunningFolder))
            np.savez("{0}/fitFeatures".format(fineTunningFolder), samples=fitFeatures, labels=fitLabels)
            np.savez("{0}/valFeatures".format(fineTunningFolder), samples=valFeatures, labels=valLabels)
            np.savez("{0}/testFeatures".format(fineTunningFolder), samples=testFeatures, labels=testLabels)
        
        # Flatten extracted features
        self._log.debug("{0} - fineTune - Flattening values to: 1*1*{1}".format(self._log_prefix, self._batch_size))
        fitFeatures = np.reshape(fitFeatures, (len(fitSamples), 1*1*self._batch_size))
        valFeatures = np.reshape(valFeatures, (len(valSamples), 1*1*self._batch_size))
        testFeatures = np.reshape(testFeatures, (len(testSamples), 1*1*self._batch_size))
        # Compose new model
        self._classifierNumClasses = len(np.unique(np.argmax(fitLabels, axis=1)))
        self.__addExtraLayers()
        # Return flatten extracted features
        return fitFeatures, valFeatures, testFeatures

    def fit(self, fitSamples, fitLabels, epochs, batch_size, validation_data, callbacks, verbose):
        self._log.debug("{0} - Fitting model with params:\n[\n\t - fit_samples = {1}\
                                                    \n\t - fit_labels = {2}\
                                                    \n\t - epochs = {3}\
                                                    \n\t - batch_size = {4}\
                                                    \n\t - val_samples = {5}\
                                                    \n\t - val_labels = {6}\
                                                    \n\t - callbacks = {7}\
                                                    \n\t - verbose = {8}\
                                                    \n]".format( self._log_prefix,
                                                                 fitSamples.shape,
                                                                 fitLabels.shape, 
                                                                 epochs,
                                                                 batch_size, 
                                                                 validation_data[0].shape, 
                                                                 validation_data[1].shape, 
                                                                 callbacks, 
                                                                 verbose ) )
        return self._modelInstance.fit( fitSamples, 
                                        fitLabels, 
                                        epochs=epochs, 
                                        batch_size=batch_size,
                                        validation_data=validation_data,
                                        callbacks=callbacks, 
                                        verbose=verbose )

    def save(self, path):
        self._modelInstance.save(path)

    def __addExtraLayers(self):
        # TODO: Provide a more elegant way to generate fine tunning models
        from keras import models
        from keras import layers
        self._classifier = self._config["options"]["custom"]["fine-tunning"]["extra_layers"]["classifier"]
        model = models.Sequential()
        model.add(layers.Dense(self._batch_size, activation='relu', input_dim=(1*1*self._batch_size)))
        model.add(layers.LeakyReLU(alpha=self._classifier["relu"]["alpha"]))
        model.add(layers.Dense(self._classifierNumClasses, activation='softmax'))
        self._modelInstance = model

    def __checkIfSuitableDataSetProperties(self):
        self.__checkNumberOfChannels()

    def __lookForPreviousExtractedFeatures(self, path):
        return os.path.exists(path)

    def __tryToLoadModelFromKeras(self):
        if self._fineTunning is True:
            self._log.debug("{0} - __tryToLoadModelFromKeras - Instanciating conv_base for fine tunning".format(self._log_prefix))
            inputShape = (self._input_height, self._input_width, self._input_depth)
            self._conv_base = getattr(self._kerasAppsModule, self._modelNameFromKeras)( include_top=False, input_shape=inputShape)
        else:
            # Model for prediction as: weights="imagenet" and include_top=True by default
            self._modelInstance = getattr(self._kerasAppsModule, self._modelNameFromKeras)()
