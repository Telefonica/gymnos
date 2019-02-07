import logging

class Flatten(object):
    def __init__(self, settings, framework):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "FLATTEN"
        self._settings = settings
        self._framework = framework
        self.__buildLayer()

    def getInstance(self):
        return self._instance

    def __buildLayer(self):
        if self._framework == "keras":
            from keras import layers
            self._instance = layers.Flatten()
            self._log.debug("{0} - Building layer with params:\n[\n\t - None \
                                                                 \n]".format( self._log_prefix ) )