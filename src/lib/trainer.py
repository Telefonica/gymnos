#
#
#   Trainer
#
#

import os
import math
import GPUtil
import cpuinfo
import platform
import logging
import numpy as np

from collections import OrderedDict
from keras.utils import to_categorical

from .trackers import History
from .services import DownloadManager
from .utils.text_utils import humanize_bytes
from .callbacks import CallbackList, TimeHistory
from .utils.data import Subset, DataLoader, get_approximate_nbytes


logger = logging.getLogger(__name__)


class Trainer:
    """
    Entrypoint to run experiment given an experiment, a model, a dataset, a training and a tracking.

    Parameters
    ----------
    dl_manager: DownloadManager
        DownloadManager to fetch data files.
    trackings_dir: str
        Directory to store tracking outputs. By default, current directory
    """

    def __init__(self, dl_manager=None, trackings_dir=None):
        if dl_manager is None:
            dl_manager = DownloadManager()
        if trackings_dir is None:
            trackings_dir = os.getcwd()

        self.dl_manager = dl_manager
        self.trackings_dir = trackings_dir

    def train(self, model, dataset, training, tracking, callbacks=None):
        """
        Run an experiment

        Parameters
        ----------
        model: core.Model
        dataset: core.Dataset
        training: core.Training
        tracking: core.Tracking
        """
        logger.info("Running experiment with id: {}".format(tracking.run_id))

        if callbacks is None:
            callbacks = []

        callbacks = CallbackList(callbacks)

        time_history = TimeHistory()

        callbacks.add(time_history)

        callbacks.on_train_begin()

        # RETRIEVE PLATFORM DETAILS

        cpu_info = cpuinfo.get_cpu_info()

        gpus_info = []
        for gpu in GPUtil.getGPUs():
            gpus_info.append({
                "name": gpu.name,
                "memory": gpu.memoryTotal
            })

        system_info = {
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "system": platform.system(),
            "node": platform.node(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "cpu": {
                "brand": cpu_info["brand"],
                "cores": cpu_info["count"]
            },
            "gpu": gpus_info
        }

        for name, key in zip(("Python version", "Platform"), ("python_version", "platform")):
            logger.debug("{}: {}".format(name, system_info[key]))

        logger.debug("Found {} GPUs".format(len(gpus_info)))

        # START TRACKING

        os.makedirs(self.trackings_dir, exist_ok=True)

        history_tracker = History()
        tracking.trackers.add(history_tracker)

        tracking.trackers.start(run_id=tracking.run_id, logdir=self.trackings_dir)

        # LOG PARAMETERS

        tracking.trackers.log_params(model.parameters)
        tracking.trackers.log_params(tracking.additional_params)
        if tracking.log_model_params:
            tracking.trackers.log_params(model.parameters)

        if tracking.log_training_params:
            tracking.trackers.log_params(training.parameters)

        for model_param_name, model_param_value in training.parameters.items():
            str_model_param_value = str(model_param_value)
            if isinstance(model_param_value, dict):
                str_model_param_value = ", ".join(name + "=" + val for name, val in model_param_value.items())

            max_line_length = 50

            if len(str_model_param_value) < max_line_length:
                logger.debug("Training with defined parameter {}: {}".format(model_param_name, str_model_param_value))
            else:
                if isinstance(model_param_value, (list, tuple)):
                    logger.debug("Training with defined parameter {} of length {}".format(model_param_name,
                                                                                          len(model_param_value)))
                else:
                    str_model_param_value = str_model_param_value[:max_line_length]
                    logger.debug("Training with defined parameter {}: {} ...".format(model_param_name,
                                                                                     str_model_param_value))

        # DOWNLOAD DATA

        callbacks.on_download_and_prepare_data_begin()

        logger.info("Downloading/preparing data")
        dataset.dataset.download_and_prepare(self.dl_manager)

        callbacks.on_download_and_prepare_data_end()

        logger.debug("Downloading/preparing data took {:.2f}s".format(time_history.times["download_and_prepare_data"]))

        # LOG DATASET PROPERTIES

        dataset_info = dataset.dataset.info()

        logger.debug("Dataset Features: {}".format(dataset_info.features))
        logger.debug("Dataset Labels: {}".format(dataset_info.labels))

        nbytes = get_approximate_nbytes(dataset.dataset)
        logger.debug("Full Dataset Samples: {}".format(len(dataset.dataset)))
        logger.debug("Full Dataset Memory Usage (approx): {}".format(humanize_bytes(nbytes)))

        # SPLIT DATASET INTO TRAIN AND TEST

        callbacks.on_train_test_split_begin()

        logger.info("Spliting dataset into train and test")

        train_samples = dataset.samples.train
        test_samples = dataset.samples.test

        if 0.0 < train_samples < 1.0:
            train_samples = math.floor(train_samples * len(dataset.dataset))
        if 0.0 < test_samples < 1.0:
            test_samples = math.floor(test_samples * len(dataset.dataset))

        logger.debug("Using {} samples for training".format(train_samples))
        logger.debug("Using {} samples for validation".format(test_samples))

        train_indices = np.arange(train_samples)
        test_indices  = np.arange(train_samples, train_samples + test_samples)

        indices = np.arange(len(dataset.dataset))

        logger.info("Shuffling dataset")

        if dataset.shuffle:
            indices = np.random.permutation(indices)

        train_indices = indices[:train_samples]
        test_indices = indices[train_samples:(train_samples + test_samples)]

        train_subset = Subset(dataset.dataset, train_indices)
        test_subset  = Subset(dataset.dataset, test_indices)

        callbacks.on_train_test_split_end()

        logger.debug("Train/test splitting took {:.2f}".format(time_history.times["train_test_split"]))

        # LOAD DATA

        callbacks.on_load_data_begin()

        logger.info("Loading data")

        load_data_by_chunks = dataset.chunk_size is not None

        if load_data_by_chunks:
            train_loader = DataLoader(train_subset, batch_size=dataset.chunk_size, drop_last=False)
            test_loader  = DataLoader(test_subset, batch_size=dataset.chunk_size, drop_last=False)
        else:
            train_data = DataLoader(train_subset, batch_size=len(dataset.dataset), drop_last=False, verbose=True)[0]
            test_data = DataLoader(test_subset, batch_size=len(dataset.dataset), drop_last=False, verbose=True)[0]

        callbacks.on_load_data_end()

        logger.debug("Loading data took {:.2f}s".format(time_history.times["load_data"]))

        # PREPROCESS DATA

        callbacks.on_fit_preprocessors_begin()

        logger.info("Fitting preprocessors using training data ({})".format(dataset.preprocessors))

        if load_data_by_chunks:
            dataset.preprocessors.fit_generator(train_loader)
        else:
            dataset.preprocessors.fit(train_data[0], train_data[1])

        callbacks.on_fit_preprocessors_end()

        logger.debug("Fitting preprocessors took {:.2f}s".format(time_history.times["fit_preprocessors"]))

        def augment_data(data):
            if not dataset.data_augmentors:
                return data

            augmented_features = []

            for index in range(len(data[0])):
                item = dataset.data_augmentors.transform(data[0][index])
                augmented_features.append(item)

            return np.array(augmented_features), data[1]

        def preprocess(data):
            """
            Preprocess batch of data (X, y):
                1. Preprocessing
                2. Convert to one-hot encoding if needed
            """
            # make sure data is a list and not a tuple (can't modify tuples)
            data = list(data)

            data[0] = dataset.preprocessors.transform(data[0])

            if dataset.one_hot:
                data[1] = to_categorical(data[1], dataset_info.labels.num_classes)

            return data

        def augment_and_preprocess(data):
            data = augment_data(data)
            data = preprocess(data)
            return data

        callbacks.on_preprocess_begin()

        logger.info("Preprocessing")

        if load_data_by_chunks:
            train_loader.transform_func = augment_and_preprocess
            test_loader.transform_func = preprocess
        else:
            train_data = augment_and_preprocess(train_data)
            test_data = preprocess(test_data)

        callbacks.on_preprocess_end()

        logger.debug("Preprocessing took {:.2f}s".format(time_history.times["preprocess"]))

        callbacks.on_preprocess_end()

        # FIT MODEL

        callbacks.on_fit_model_begin()

        logger.info("Fitting model")

        if load_data_by_chunks:
            train_metrics = model.model.fit_generator(train_loader, **training.parameters)
        else:
            train_metrics = model.model.fit(train_data[0], train_data[1], **training.parameters)

        callbacks.on_fit_model_end()

        logger.debug("Fitting model took {:.2f}s".format(time_history.times["fit_model"]))

        for metric_name, metric_value in train_metrics.items():
            logger.info("Results for {} -> mean={:.2f}, min={:.2f}, max={:.2f}".format(metric_name,
                                                                                       np.mean(metric_value),
                                                                                       np.min(metric_value),
                                                                                       np.max(metric_value)))
        if tracking.log_model_metrics:
            tracking.trackers.log_metrics(train_metrics)

        # EVALUATE MODEL

        callbacks.on_evaluate_model_begin()

        logger.info("Evaluating model using test data")

        if load_data_by_chunks:
            test_metrics = model.model.evaluate_generator(test_loader)
        else:
            test_metrics = model.model.evaluate(test_data[0], test_data[1])

        callbacks.on_evaluate_model_end()

        logger.debug("Evaluating model took {:.2f}s".format(time_history.times["evaluate_model"]))

        for metric_name, metric_value in test_metrics.items():
            logger.info("test_{}={}".format(metric_name, metric_value))

        if tracking.log_model_metrics:
            tracking.trackers.log_metrics(test_metrics, prefix="test_")

        tracking.trackers.end()

        callbacks.on_train_end()

        return OrderedDict([
            ["run_id", tracking.run_id],
            ["start_datetime", history_tracker.start_datetime.timestamp()],
            ["end_datetime", history_tracker.end_datetime.timestamp()],
            ["elapsed_times", time_history.times],
            ["tags", history_tracker.tags],
            ["params", history_tracker.params],
            ["metrics", history_tracker.metrics],
            ["system_info", system_info],
            ["trackings_dir", self.trackings_dir],
            ["download_dir", self.dl_manager.download_dir],
            ["extract_dir", self.dl_manager.extract_dir]
        ])
