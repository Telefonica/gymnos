import logging

class Convolutional2D(object):
    def __init__(self, settings, framework):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "CONV_2D"
        self._settings = settings
        self._framework = framework
        self._inputShape = settings["input_shape"] if "input_shape" in settings else None
        self.__buildLayer()

    def getInstance(self):
        return self._instance

    def __buildLayer(self):
        if self._framework == "keras":
            myFilter = self._settings["filter"]
            kernelSize = self._settings["kernel_size"]
            activation = self._settings["activation"]
            from keras import layers
            if self._inputShape: 
                self._instance = layers.Conv2D( myFilter, kernelSize, activation=activation, input_shape=self._inputShape )
            else:
                self._instance = layers.Conv2D( myFilter, kernelSize, activation=activation)
            self._log.debug("{0} - Building layer with params:\n[\n\t - filter = {1}\
                                                                 \n\t - kernel_size = {2}\
                                                                 \n\t - activation = {3}\
                                                                 \n\t - input_shape = {4}\
                                                                 \n]".format( self._log_prefix,
                                                                              myFilter,
                                                                              kernelSize, 
                                                                              activation,
                                                                              self._inputShape ) )
