#
#
#   Trainer
#
#

import os
import dill
import GPUtil
import cpuinfo
import platform
import logging
import tempfile
import numpy as np

from collections import OrderedDict

from .utils.py_utils import chain
from .trackers.history import History
from .utils.np_utils import to_categorical
from .utils.text_utils import humanize_bytes
from .utils.archiver import extract_zip, zipdir
from .callbacks import CallbackList, TimeHistory
from .services.download_manager import DownloadManager
from .core.model import Model
from .core.dataset import Dataset
from .core.tracking import Tracking
from .datasets.dataset import IterableDataset
from .utils.data import DataLoader, IterableDataLoader, split_sequence, split_iterator

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

    def __init__(self, model, dataset, tracking=None):
        if tracking is None:
            tracking = Tracking()

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
            Dictionnary with the following required keys:

                - ``"model"``
                - ``"dataset"``

            And optionally the following keys:

                - ``"tracking"``

        Returns
        -------
        trainer: Trainer
        """
        model = spec["model"]
        dataset = spec["dataset"]
        tracking = spec.get("tracking", {})  # optional property
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

        # SPLIT DATASET INTO TRAIN AND TEST AND PREPARE / LOAD DATA

        callbacks.on_load_data_begin()

        should_load_data_by_chunks = self.dataset.chunk_size is not None
        is_iterable_dataset = isinstance(self.dataset.dataset, IterableDataset)

        if should_load_data_by_chunks:
            logger.info("Preparing generator to load data")
        else:
            logger.info("Loading data into memory")

        split_func = split_iterator if is_iterable_dataset else split_sequence

        train_dataset, test_dataset = split_func(self.dataset.dataset, (self.dataset.samples.train,
                                                                        self.dataset.samples.test),
                                                 shuffle=self.dataset.shuffle)

        data_loader_cls = IterableDataLoader if is_iterable_dataset else DataLoader

        if should_load_data_by_chunks:
            train_dataset = data_loader_cls(train_dataset, batch_size=self.dataset.chunk_size, drop_last=False)
            test_dataset = data_loader_cls(test_dataset, batch_size=self.dataset.chunk_size, drop_last=False)
        else:
            train_dataset = next(iter(data_loader_cls(train_dataset, batch_size=len(train_dataset),
                                                      drop_last=False, verbose=True)))
            test_dataset = next(iter(data_loader_cls(test_dataset, batch_size=len(test_dataset),
                                                     drop_last=False, verbose=True)))

        callbacks.on_load_data_end()

        logger.debug("Preparing / loading data took {:.2f}s".format(time_history.times["load_data"]))

        # PREPROCESS DATA

        callbacks.on_fit_preprocessors_begin()

        logger.info("Fitting preprocessors using training data ({})".format(self.dataset.preprocessors))

        if should_load_data_by_chunks:
            self.dataset.preprocessors.fit_generator(train_dataset)
        else:
            self.dataset.preprocessors.fit(*train_dataset)

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

            Parameters
            -----------
            data: tuple or list
                Tuple of X, y
            """
            # make sure data is a list and not a tuple (can't modify tuples)
            data = list(data)

            data[0] = self.dataset.preprocessors.transform(data[0])

            if self.dataset.one_hot:
                data[1] = to_categorical(data[1], self.dataset.dataset.labels_info.num_classes)

            return data

        callbacks.on_preprocess_begin()

        logger.info("Preprocessing")

        if should_load_data_by_chunks:
            train_dataset.transform_func = chain(augment, preprocess)
            test_dataset.transform_func = preprocess
        else:
            train_dataset = chain(augment, preprocess)(train_dataset)
            test_dataset = preprocess(test_dataset)

        callbacks.on_preprocess_end()

        logger.debug("Preprocessing took {:.2f}s".format(time_history.times["preprocess"]))

        # FIT MODEL

        callbacks.on_fit_model_begin()

        logger.info("Fitting model")

        if should_load_data_by_chunks:
            train_metrics = self.model.model.fit_generator(train_dataset, **self.model.training)
        else:
            train_metrics = self.model.model.fit(*train_dataset, **self.model.training)

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

        if should_load_data_by_chunks:
            test_metrics = self.model.model.evaluate_generator(test_dataset)
        else:
            test_metrics = self.model.model.evaluate(*test_dataset)

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

        logger.info("Predicting labels")
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

        logger.info("Predicting label probabilities")
        probs = self.model.model.predict_proba(X)

        return probs

    def save(self, path):
        """
        Save trainer instance.

        Parameters
        ----------
        path: str
            Path to store trainer (zipped file)
        """
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "model.pkl"), "wb") as fp:
                dill.dump(self.model.to_dict(), fp)

            with open(os.path.join(tempdir, "tracking.pkl"), "wb") as fp:
                dill.dump(self.tracking.to_dict(), fp)

            with open(os.path.join(tempdir, "dataset.pkl"), "wb") as fp:
                dill.dump(self.dataset.to_dict(), fp)

            with open(os.path.join(tempdir, "features_info.pkl"), "wb") as fp:
                dill.dump(self.dataset.dataset.features_info, fp)

            with open(os.path.join(tempdir, "labels_info.pkl"), "wb") as fp:
                dill.dump(self.dataset.dataset.labels_info, fp)

            with open(os.path.join(tempdir, "saved_preprocessors.pkl"), "wb") as fp:
                dill.dump(self.dataset.preprocessors, fp)

            model_store_dir = os.path.join(tempdir, "saved_model")
            os.makedirs(model_store_dir)
            self.model.model.save(model_store_dir)

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
                model = Model(**dill.load(fp))

            with open(os.path.join(tempdir, "dataset.pkl"), "rb") as fp:
                dataset = Dataset(**dill.load(fp))

            with open(os.path.join(tempdir, "tracking.pkl"), "rb") as fp:
                tracking = Tracking(**dill.load(fp))

            model.model.restore(os.path.join(tempdir, "saved_model"))

            with open(os.path.join(tempdir, "features_info.pkl"), "rb") as fp:
                features_info = dill.load(fp)

            with open(os.path.join(tempdir, "labels_info.pkl"), "rb") as fp:
                labels_info = dill.load(fp)

            with open(os.path.join(tempdir, "saved_preprocessors.pkl"), "rb") as fp:
                dataset.preprocessors = dill.load(fp)

            # assign saved @property
            cls = type(dataset.dataset)
            cls = type(cls.__name__, (cls,), {})
            dataset.dataset.__class__ = cls
            setattr(cls, "features_info", features_info)
            setattr(cls, "labels_info", labels_info)

        return Trainer(model, dataset, tracking)
