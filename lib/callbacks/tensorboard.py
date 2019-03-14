from keras import callbacks

from lib.log import logger
from . import callback


class TensorBoard(callback.Callback):
    def __init__(self, config, runTimeConfig):
        super().__init__()

        self._log = logger.get_logger()
        self._log_prefix = logger.setup_prefix(__class__)
        self._config = config
        self._runTimeConfig = runTimeConfig
        self._log_dir = "{0}/{1}".format(runTimeConfig["train_dir"],
                                         'logs-tensorboard') if "train_dir" in runTimeConfig else './logs'
        self._histogram_freq = config["histogram_freq"] if "histogram_freq" in config else 0
        self._batch_size = config["batch_size"] if "monitbatch_sizeor" in config else 32
        self._write_graph = config["write_graph"] if "write_graph" in config else True
        self._write_grads = config["write_grads"] if "write_grads" in config else False
        self._write_images = config["write_images"] if "write_images" in config else False
        self._embeddings_freq = config["embeddings_freq"] if "embeddings_freq" in config else 0
        self._embeddings_layer_names = config["embeddings_layer_names"] if "embeddings_layer_names" in config else None
        self._embeddings_metadata = config["embeddings_metadata"] if "embeddings_metadata" in config else None
        self._embeddings_data = config["embeddings_data"] if "embeddings_data" in config else None
        self._update_freq = config["update_freq"] if "update_freq" in config else 'epoch'
        self.__buildCallback()

    def getInstance(self):
        return self._instance

    def __buildCallback(self):
        # Only Keras support so far
        self._log.debug("{0} - Instance with params:\n[\n\t - log_dir = {1}\
                        \n\t - histogram_freq = {2}\
                        \n\t - batch_size = {3}\
                        \n\t - write_graph = {4}\
                        \n\t - write_grads = {5}\
                        \n\t - write_images = {6}\
                        \n\t - embeddings_freq = {7}\
                        \n\t - embeddings_layer_names = {8}\
                        \n\t - embeddings_metadata = {9}\
                        \n\t - embeddings_data = {10}\
                        \n\t - update_freq = {11}\
                        \n]".format(self._log_prefix,
                                    self._log_dir,
                                    self._histogram_freq,
                                    self._batch_size,
                                    self._write_graph,
                                    self._write_grads,
                                    self._write_images,
                                    self._embeddings_freq,
                                    self._embeddings_layer_names,
                                    self._embeddings_metadata,
                                    self._embeddings_data,
                                    self._update_freq))
        self._instance = callbacks.TensorBoard(log_dir=self._log_dir,
                                               histogram_freq=self._histogram_freq,
                                               batch_size=self._batch_size,
                                               write_graph=self._write_graph,
                                               write_grads=self._write_grads,
                                               write_images=self._write_images,
                                               embeddings_freq=self._embeddings_freq,
                                               embeddings_layer_names=self._embeddings_layer_names,
                                               embeddings_metadata=self._embeddings_metadata,
                                               embeddings_data=self._embeddings_data,
                                               update_freq=self._update_freq)
