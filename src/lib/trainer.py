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
from .datasets import HDF5Dataset
from .utils.timing import ElapsedTimeCalculator
from .services import DownloadManager
from .utils.text_utils import humanize_bytes
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

    def train(self, experiment, model, dataset, training, tracking):
        """
        Run an experiment

        Parameters
        ----------
        experiment: core.Experiment
        model: core.Model
        dataset: core.Dataset
        training: core.Training
        tracking: core.Tracking
        """
        logger.info("Starting experiment {}".format(experiment.name))

        elapsed_time_calc = ElapsedTimeCalculator()

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

        tracking.trackers.start(run_name=experiment.name, logdir=self.trackings_dir)

        # LOG PARAMETERS

        tracking.trackers.log_params(model.parameters)
        tracking.trackers.log_params(tracking.additional_params)

        # DOWNLOAD DATA

        logger.info("Downloading and preparing data")
        with elapsed_time_calc("dataset_download_and_prepare") as elapsed:
            dataset.dataset.download_and_prepare(self.dl_manager)

        logger.debug("Downloading and preparing data took {:.2f}".format(elapsed.s))

        # LOG DATASET PROPERTIES

        dataset_info = dataset.dataset.info()

        logger.debug("Dataset Features: {}".format(dataset_info.features))
        logger.debug("Dataset Labels: {}".format(dataset_info.labels))

        nbytes = get_approximate_nbytes(dataset.dataset)
        logger.debug("Full Dataset Samples: {}".format(len(dataset.dataset)))
        logger.debug("Full Dataset Memory Usage (approx): {}".format(humanize_bytes(nbytes)))

        if isinstance(dataset.dataset, HDF5Dataset):
            logger.debug("Using HDF5 dataset")

        # SPLIT DATASET INTO TRAIN AND TEST

        logger.info("Spliting dataset into train and test")

        train_samples = dataset.samples.train
        test_samples = dataset.samples.test

        if 0.0 < train_samples < 1.0:
            train_samples = math.floor(train_samples * len(dataset.dataset))
        if 0.0 < test_samples < 1.0:
            test_samples = math.floor(test_samples * len(dataset.dataset))

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

        # LOAD DATA

        load_data_by_chunks = dataset.chunk_size is not None

        if load_data_by_chunks:
            train_loader = DataLoader(train_subset, batch_size=dataset.chunk_size, drop_last=False)
            test_loader  = DataLoader(test_subset, batch_size=dataset.chunk_size, drop_last=False)
        else:
            logger.info("Loading data into memory")

            with elapsed_time_calc("load_data") as elapsed:
                train_data = DataLoader(train_subset, batch_size=len(dataset.dataset), drop_last=False, verbose=True)[0]
                test_data = DataLoader(test_subset, batch_size=len(dataset.dataset), drop_last=False, verbose=True)[0]

            logger.debug("Loading data into memory took {:.2f}s".format(elapsed.s))

        # PREPROCESS DATA

        logger.info("Fitting preprocessing pipeline using training data ({})".format(dataset.preprocessors))

        with elapsed_time_calc("fit_preprocessors") as elapsed:
            if load_data_by_chunks:
                dataset.preprocessors.fit_generator(train_loader)
            else:
                dataset.preprocessors.fit(train_data[0], train_data[1])

        logger.debug("Fitting preprocessing pipeline took {:.2f}s".format(elapsed.s))

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

        if load_data_by_chunks:
            train_loader.transform_func = augment_and_preprocess
            test_loader.transform_func = preprocess
        else:
            logger.info("Preprocessing data ({})".format(dataset.preprocessors))

            with elapsed_time_calc("transform_data") as elapsed:
                train_data = augment_and_preprocess(train_data)
                test_data = preprocess(test_data)

            logger.debug("Preprocessing data took {:.2f}s".format(elapsed.s))

        # FIT MODEL

        logger.info("Fitting model using training data")

        with elapsed_time_calc("fit_model") as elapsed:
            if load_data_by_chunks:
                train_metrics = model.model.fit_generator(train_loader, **training.parameters)
            else:
                train_metrics = model.model.fit(train_data[0], train_data[1], **training.parameters)

        logger.debug("Fitting model took {:.2f}s".format(elapsed.s))

        for metric_name, metric_value in train_metrics.items():
            logger.info("Results for {} -> mean={:.2f}, min={:.2f}, max={:.2f}".format(metric_name,
                                                                                       np.mean(metric_value),
                                                                                       np.min(metric_value),
                                                                                       np.max(metric_value)))
        tracking.trackers.log_metrics(train_metrics)

        # EVALUATE MODEL

        logger.info("Evaluating model using test data")

        with elapsed_time_calc("evaluate_model") as elapsed:
            if load_data_by_chunks:
                test_metrics = model.model.evaluate_generator(test_loader)
            else:
                test_metrics = model.model.evaluate(test_data[0], test_data[1])

        logger.debug("Evaluating model took {:.2f}s".format(elapsed.s))

        for metric_name, metric_value in test_metrics.items():
            logger.info("test_{}={}".format(metric_name, metric_value))

        tracking.trackers.log_metrics(test_metrics, prefix="test_")

        tracking.trackers.end()

        return OrderedDict([
            ["experiment_name", experiment.name],
            ["start_datetime", history_tracker.start_datetime.timestamp()],
            ["end_datetime", history_tracker.end_datetime.timestamp()],
            ["elapsed_times", elapsed_time_calc.times],
            ["tags", history_tracker.tags],
            ["params", history_tracker.params],
            ["metrics", history_tracker.metrics],
            ["system_info", system_info],
            ["trackings_dir", self.trackings_dir],
            ["execution_dir", os.getcwd()],
            ["download_dir", self.dl_manager.download_dir],
            ["extract_dir", self.dl_manager.extract_dir]
        ])
