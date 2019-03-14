from lib.log import logger
from . import model
from .utils.layers.layer_factory import LayerFactory


class CustomStack(model.Model):

    def __init__(self, config):
        model.Model.__init__(self, config)
        self._log = logger.get_logger()
        self._log_prefix = logger.setup_prefix(__class__)
        self._modelInstance = None
        self._config = config
        self._modelId = config["id"]
        self._optimizer = config["options"]["compilation"]["optimizer"]
        self._loss = config["options"]["compilation"]["loss"]
        self._metrics = config["options"]["compilation"]["metrics"]
        self._framework = config["options"]["custom"]["framework"]
        self._layers = self._config["options"]["custom"]["stack"]["layers"]
        self._lf = LayerFactory(self._framework)

    def compile(self):
        # TODO: Check all parameters are included, otherwise launch assert or protect code
        if self._modelInstance:
            self._log.info("{0} - Compiling model with:\n[\n\t - loss = {1}" +
                           "\n\t - optimizer = {2}" +
                           "\n\t - metrics = {3}" +
                           "\n]".format(self._log_prefix,
                                        self._loss,
                                        self._optimizer,
                                        self._metrics))
            self._modelInstance.compile(loss=self._loss,
                                        optimizer=self._optimizer,
                                        metrics=self._metrics)
        else:
            errMsg = "{0} - Model can not be compiled. Model instance does not exist".format(self._log_prefix)
            self._log.error(errMsg)
            raise ValueError(errMsg)

    def evaluate(self, samples, labels):
        return self._modelInstance.evaluate(samples, labels)

    def init(self):
        self._log.debug("{0} - init - Building model stack from config...".format(self._log_prefix))
        if self._framework  == "keras":
            from keras import models
            self._modelInstance = models.Sequential()
            for layer in self._layers:
                self._modelInstance.add(self._lf.factory(layer))

    def fineTune(self, fitSamples, valSamples, testSamples, fitLabels, valLabels, testLabels, trainDir):
        pass

    def fit(self, fitSamples, fitLabels, epochs, batch_size, validation_data, callbacks, verbose):
        self._log.debug("{0} - Fitting model with params:\n[\n\t - fit_samples = {1}\
                        \n\t - fit_labels = {2}\
                        \n\t - epochs = {3}\
                        \n\t - batch_size = {4}\
                        \n\t - val_samples = {5}\
                        \n\t - val_labels = {6}\
                        \n\t - callbacks = {7}\
                        \n\t - verbose = {8}\
                        \n]".format(self._log_prefix,
                                    fitSamples.shape,
                                    fitLabels.shape,
                                    epochs,
                                    batch_size,
                                    validation_data[0].shape,
                                    validation_data[1].shape,
                                    callbacks,
                                    verbose))
        return self._modelInstance.fit(fitSamples,
                                       fitLabels,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_data=validation_data,
                                       callbacks=callbacks,
                                       verbose=verbose)

    def save(self, path):
        self._modelInstance.save(path)
