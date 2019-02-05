import logging

class Dense(object):
    def __init__(self, settings, framework):
        self._log = logging.getLogger('gymnosd')
        self._log_prefix = "DENSE"
        self._settings = settings
        self._framework = framework
        self.__buildLayer()

    def __buildLayer(self):
        if self._framework == "keras":
            from keras import layers
            units = self._settings["units"]
            activation = self._settings["activation"]
            layers.Dense( units, activation )
        self._log.debug("{0} - Building layer with params:\n[\n\t - units = {1}\
                                                             \n\t - kernel_size = {2}\
                                                             \n]".format( self._log_prefix, 
                                                                          units, 
                                                                          activation ) )