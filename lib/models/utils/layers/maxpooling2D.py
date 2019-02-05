import logging

class MaxPooling2D(object):
    def __init__(self, settings, framework):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "MAXPOOLING_2D"
        self._settings = settings
        self._framework = framework
        self.__buildLayer()

    def __buildLayer(self):
        if self._framework == "keras":
            poolSize = self._settings["pool_size"]
            from keras import layers
            layers.MaxPooling2D( poolSize )
            self._log.debug("{0} - Building layer with params:\n[\n\t - pool_size = {1}\
                                                                 \n]".format( self._log_prefix, 
                                                                              poolSize ) )