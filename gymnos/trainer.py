#
#
#   Trainer
#
#

import os
import dill
import math
import GPUtil
import cpuinfo
import platform
import logging
import tempfile
import numpy as np

from collections import OrderedDict
from tensorflow.keras.utils import to_categorical

from . import models, datasets
from .trackers.history import History
from .trackers.tracker import TrackerList
from .services.download_manager import DownloadManager
from .utils.data import Subset, DataLoader
from .utils.text_utils import humanize_bytes
from .utils.archiver import extract_zip, zipdir
from .callbacks import CallbackList, TimeHistory
from .core.model import Model
from .core.dataset import Dataset
from .core.tracking import Tracking

logger = logging.getLogger(__name__)


class Trainer:
    """
    Entrypoint to run experiment given an experiment, a model, a dataset, a training and a tracking.

    Parameters
    ----------
    model: gymnos.core.model.Model
        Instance of core model
    dataset: gymnos.core.dataset.Dataset
        Instance of core.dataset
    tracking: gymnos.core.tracking.Tracking
        Instance of core.tracking
    """

    def __init__(self, model, dataset, tracking):
        self.model = model
        self.dataset = dataset
        self.tracking = tracking

    @staticmethod
    def from_dict(spec):
        """
        Create trainer from dictionnary specifying experiment, model, dataset, training and tracking specs.

        Parameters
        ----------
        spec: dict
            Dictionnary with the following keys:

                - ``"model"``
                - ``"dataset"``
                - ``"tracking"``

        Returns
        -------
        trainer: Trainer
        """
        model = spec.get("model", {})
        dataset = spec.get("dataset", {})
        tracking = spec.get("tracking", {})
        return Trainer(
            model=Model(**model),
            dataset=Dataset(**dataset),
            tracking=Tracking(**tracking)
        )

    def to_dict(self):
        """
        Convert trainer to dictionnary specifying core components

        Returns
        -----------
        dict
            Trainer components
        """
        return dict(
            dataset=self.dataset.to_dict(),
            model=self.model.to_dict(),
            tracking=self.tracking.to_dict()
        )

    def train(self, dl_manager=None, trackings_dir=None, callbacks=None):
        """
        Run trainer.

        Parameters
        ----------
        dl_manager: DownloadManager
            DownloadManager to fetch data files.
        trackings_dir: str
            Directory to store tracking outputs. By default, current directory
        callbacks: list of Callback
            Callbacks to run during training

        Returns
        -------
        results: dict
            Training results with the following keys: "experiment_name", "start_datetime", "end_datetime",
            "elapsed_times", "tags", "params", "metrics", "system_info", "trackings_dir", "download_dir",
            "extract_dir",
        """

        if dl_manager is None:
            dl_manager = DownloadManager()
        if trackings_dir is None:
            trackings_dir = os.getcwd()

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

        os.makedirs(trackings_dir, exist_ok=True)

        history_tracker = History()
        self.tracking.trackers.add(history_tracker)

        self.tracking.trackers.start(run_id=self.tracking.run_id, logdir=trackings_dir)

        # LOG PARAMETERS

        if self.tracking.log_model_params:
            self.tracking.trackers.log_params(self.model.parameters)

        if self.tracking.log_training_params:
            self.tracking.trackers.log_params(self.model.training)

        self.tracking.trackers.log_tags(self.tracking.tags)

        for model_param_name, model_param_value in self.model.training.items():
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
        self.dataset.dataset.download_and_prepare(dl_manager)

        callbacks.on_download_and_prepare_data_end()

        logger.debug("Downloading/preparing data took {:.2f}s".format(time_history.times["download_and_prepare_data"]))

        # LOG DATASET PROPERTIES

        logger.debug("Dataset Features: {}".format(self.dataset.dataset.features_info))
        logger.debug("Dataset Labels: {}".format(self.dataset.dataset.labels_info))

        logger.debug("Full Dataset Samples: {}".format(len(self.dataset.dataset)))
        logger.debug("Full Dataset Memory Usage (approx): {}".format(humanize_bytes(self.dataset.dataset.nbytes)))

        # SPLIT DATASET INTO TRAIN AND TEST

        callbacks.on_train_test_split_begin()

        logger.info("Spliting dataset into train and test")

        train_samples = self.dataset.samples.train
        test_samples = self.dataset.samples.test

        if 0.0 < train_samples < 1.0:
            train_samples = math.floor(train_samples * len(self.dataset.dataset))
        if 0.0 < test_samples < 1.0:
            test_samples = math.floor(test_samples * len(self.dataset.dataset))

        logger.debug("Using {} samples for training".format(train_samples))
        logger.debug("Using {} samples for testing".format(test_samples))

        train_indices = np.arange(train_samples)
        test_indices  = np.arange(train_samples, train_samples + test_samples)

        indices = np.arange(len(self.dataset.dataset))

        logger.info("Shuffling dataset")

        if self.dataset.shuffle:
            indices = np.random.permutation(indices)

        train_indices = indices[:train_samples]
        test_indices = indices[train_samples:(train_samples + test_samples)]

        train_subset = Subset(self.dataset.dataset, train_indices)
        test_subset  = Subset(self.dataset.dataset, test_indices)

        callbacks.on_train_test_split_end()

        logger.debug("Train/test splitting took {:.2f}".format(time_history.times["train_test_split"]))

        # LOAD DATA

        callbacks.on_load_data_begin()

        logger.info("Loading data")

        load_data_by_chunks = self.dataset.chunk_size is not None

        if load_data_by_chunks:
            train_loader = DataLoader(train_subset, batch_size=self.dataset.chunk_size, drop_last=False)
            test_loader  = DataLoader(test_subset, batch_size=self.dataset.chunk_size, drop_last=False)
        else:
            train_data = DataLoader(train_subset, batch_size=len(self.dataset.dataset), drop_last=False,
                                    verbose=True)[0]
            test_data = DataLoader(test_subset, batch_size=len(self.dataset.dataset), drop_last=False,
                                   verbose=True)[0]

        callbacks.on_load_data_end()

        logger.debug("Loading data took {:.2f}s".format(time_history.times["load_data"]))

        # PREPROCESS DATA

        callbacks.on_fit_preprocessors_begin()

        logger.info("Fitting preprocessors using training data ({})".format(self.dataset.preprocessors))

        if load_data_by_chunks:
            self.dataset.preprocessors.fit_generator(train_loader)
        else:
            self.dataset.preprocessors.fit(train_data[0], train_data[1])

        callbacks.on_fit_preprocessors_end()

        logger.debug("Fitting preprocessors took {:.2f}s".format(time_history.times["fit_preprocessors"]))

        def augment(data):
            if not self.dataset.data_augmentors:
                return data

            augmented_features = []

            for index in range(len(data[0])):
                item = self.dataset.data_augmentors.transform(data[0][index])
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

            data[0] = self.dataset.preprocessors.transform(data[0])

            if self.dataset.one_hot:
                data[1] = to_categorical(data[1], self.dataset.dataset.labels_info.num_classes)

            return data

        def augment_and_preprocess(data):
            data = augment(data)
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
            train_metrics = self.model.model.fit_generator(train_loader, **self.model.training)
        else:
            train_metrics = self.model.model.fit(train_data[0], train_data[1], **self.model.training)

        callbacks.on_fit_model_end()

        logger.debug("Fitting model took {:.2f}s".format(time_history.times["fit_model"]))

        for metric_name, metric_value in train_metrics.items():
            logger.info("Results for {} -> mean={:.2f}, min={:.2f}, max={:.2f}".format(metric_name,
                                                                                       np.mean(metric_value),
                                                                                       np.min(metric_value),
                                                                                       np.max(metric_value)))
        if self.tracking.log_model_metrics:
            self.tracking.trackers.log_metrics(train_metrics)

        # EVALUATE MODEL

        callbacks.on_evaluate_model_begin()

        logger.info("Evaluating model using test data")

        if load_data_by_chunks:
            test_metrics = self.model.model.evaluate_generator(test_loader)
        else:
            test_metrics = self.model.model.evaluate(test_data[0], test_data[1])

        callbacks.on_evaluate_model_end()

        logger.debug("Evaluating model took {:.2f}s".format(time_history.times["evaluate_model"]))

        for metric_name, metric_value in test_metrics.items():
            logger.info("test_{}={}".format(metric_name, metric_value))

        if self.tracking.log_model_metrics:
            self.tracking.trackers.log_metrics(test_metrics, prefix="test_")

        self.tracking.trackers.end()

        callbacks.on_train_end()

        return OrderedDict([
            ["run_id", self.tracking.run_id],
            ["start_datetime", history_tracker.start_datetime.timestamp()],
            ["end_datetime", history_tracker.end_datetime.timestamp()],
            ["elapsed_times", time_history.times],
            ["tags", history_tracker.tags],
            ["params", history_tracker.params],
            ["metrics", history_tracker.metrics],
            ["system_info", system_info],
            ["trackings_dir", trackings_dir],
            ["download_dir", dl_manager.download_dir],
            ["extract_dir", dl_manager.extract_dir]
        ])

    def predict(self, X):
        """
        Predict values.

        Parameters
        ----------
        X: np.ndarray
            Samples to predict

        Returns
        -------
        predictions: np.ndarray
            Predicted values
        """
        logger.info("Preprocessing samples ({})".format(self.dataset.preprocessors))
        X = self.dataset.preprocessors.transform(X)

        logger.info("Predicting labels using {}".format(self.model.name))
        predictions = self.model.model.predict(X)

        return predictions

    def predict_proba(self, X):
        """
        Predict probabilities (only if model supports probabilities i.e ``predict_proba`` method is implemented)

        Parameters
        ----------
        X: np.ndarray
            Samples to predict

        Returns
        -------
        probabilities: np.ndarray
            Predicted probabilities.
        """
        logger.info("Preprocessing samples ({})".format(self.dataset.preprocessors))
        X = self.dataset.preprocessors.transform(X)

        logger.info("Predicting label probabilities using {}".format(self.model.name))
        probs = self.model.model.predict_proba(X)

        return probs

    def _save_model(self, path):
        """
        Save model without inner model instance

        Parameters
        ----------
        path: str
            Path to store pickled model
        """
        model_model = self.model.model
        self.model.model = None
        with open(path, "wb") as fp:
            dill.dump(self.model, fp)
        self.model.model = model_model

    def _save_tracking(self, path):
        """
        Save tracking without trackers

        Parameters
        ----------
        path: str
            Path to store pickled tracking
        """
        tracking_trackers = self.tracking.trackers
        self.tracking.trackers = None
        with open(path, "wb") as fp:
            dill.dump(self.tracking, fp)
        self.tracking.trackers = tracking_trackers

    def _save_dataset(self, path):
        """
        Save dataset without inner dataset and preprocessors instance

        Parameters
        ----------
        path: str
            Path to store pickled dataset
        """
        dataset_preprocessors = self.dataset.preprocessors
        self.dataset.preprocessors = None
        dataset_dataset = self.dataset.dataset
        self.dataset.dataset = None

        with open(path, "wb") as fp:
            dill.dump(self.dataset, fp)

        self.dataset.preprocessors = dataset_preprocessors
        self.dataset.dataset = dataset_dataset

    def save(self, path):
        """
        Save trainer instance.

        Parameters
        ----------
        path: str
            Path to store trainer (zipped file)
        """
        with tempfile.TemporaryDirectory() as tempdir:
            self._save_model(os.path.join(tempdir, "model.pkl"))

            model_store_dir = os.path.join(tempdir, "saved_model")
            os.makedirs(model_store_dir)
            self.model.model.save(model_store_dir)

            self._save_tracking(os.path.join(tempdir, "tracking.pkl"))

            self._save_dataset(os.path.join(tempdir, "dataset.pkl"))

            with open(os.path.join(tempdir, "saved_preprocessors.pkl"), "wb") as fp:
                dill.dump(self.dataset.preprocessors, fp)

            zipdir(tempdir, path)

    @staticmethod
    def load(path):
        """
        Load trainer instance

        Parameters
        ----------
        path: str
            Path to load trainer (zipped file)

        Returns
        -------
        Trainer
            Restored trainer instance.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            extract_zip(path, extract_dir=tempdir)

            with open(os.path.join(tempdir, "model.pkl"), "rb") as fp:
                model = dill.load(fp)
            model.model = models.load(model.name, **model.parameters)
            model.model.restore(os.path.join(tempdir, "saved_model"))

            with open(os.path.join(tempdir, "dataset.pkl"), "rb") as fp:
                dataset = dill.load(fp)
            dataset.dataset = datasets.load(dataset.name)

            with open(os.path.join(tempdir, "saved_preprocessors.pkl"), "rb") as fp:
                dataset.preprocessors = dill.load(fp)

            with open(os.path.join(tempdir, "tracking.pkl"), "rb") as fp:
                tracking = dill.load(fp)
            tracking.trackers = TrackerList.from_dict(tracking.trackers_spec)

        return Trainer(model, dataset, tracking)
