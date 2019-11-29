#
#
#   Trainer
#
#

import os
import dill
import logging
import warnings
import platform
import tempfile

from abc import ABCMeta, abstractmethod

from . import core

from .utils.py_utils import chain
from .models.model import SparkModel
from .trackers.history import History
from .datasets.dataset import SparkDataset
from .utils.np_utils import label_binarize
from .utils.archiver import zipdir, extract_zip
from .services.download_manager import DownloadManager
from .callbacks import CallbackList, TimeHistory, Logger
from .utils.hardware_info import get_cpu_info, get_gpus_info
from .utils.data import (split_spark_dataframe, split_iterator, split_sequence,
                         IterableDataLoader, DataLoader, is_sequence)


logger = logging.getLogger(__name__)


# MARK: Training Runners


class TrainingRunner(metaclass=ABCMeta):
    """
    Abstract training runner
    """

    def __init__(self, model: core.Model, dataset: core.Dataset):
        self.model = model
        self.dataset = dataset

        logger.info("Using {} for training".format(self.__class__.__name__))

    @abstractmethod
    def load_dataset(self):
        """
        Load and return dataset.

        Returns
        -----------
        any
            Dataset
        """

    @abstractmethod
    def split_dataset(self, dataset):
        """
        Split dataset into train and test according to ``self.dataset.samples``
        Parameters
        ----------
        dataset:
            Output from ``load_dataset``
        Returns
        --------
        any
            Train dataset
        any
            Test dataset
        """

    @abstractmethod
    def fit_preprocessors(self, dataset):
        """
        Fit ``dataset.preprocessors`` to dataset
        """

    @abstractmethod
    def transform_dataset(self, dataset):
        """
        Transform dataset using ``dataset.preprocessors``.

        Returns
        --------
        any
            Transformed_dataset
        """

    @abstractmethod
    def augment_dataset(self, dataset):
        """
        Augment dataset using ``dataset.data_augmentors``

        Returns
        --------
        any
            Augmented dataset
        """

    @abstractmethod
    def one_hot_encode(self, dataset):
        """
        One hot encode labels.
        Returns
        -------
        any
            Dataset with one-hot encoded labels
        """

    @abstractmethod
    def fit_model(self, dataset):
        """
        Fit ``model.model`` to dataset.

        Returns
        --------
        metrics: dict
            Fit metrics
        """

    @abstractmethod
    def evaluate_model(self, dataset):
        """
        Evualate ``model.model`` according to dataset.
        Returns
        --------
        metrics: dict
            Evaluation metrics
        """


class InMemoryTrainingRunner(TrainingRunner):
    """
    Runner for datasets fully loaded into memory
    """

    def __init__(self, model: core.Model, dataset: core.Dataset):
        super().__init__(model, dataset)

        logger.info("Data will be loaded into memory")

    def load_dataset(self):
        logger.info("To prevent rows of a dataset that are not going to be used from being loaded into memory, "
                    "the loading into memory will be performed when we split the dataset")
        return self.dataset.dataset

    def split_dataset(self, dataset):
        if is_sequence(dataset):
            split_func = split_sequence
        else:
            split_func = split_iterator

        (train_dataset, test_dataset) = split_func(dataset, (self.dataset.samples.train, self.dataset.samples.test),
                                                   shuffle=self.dataset.shuffle, random_state=self.dataset.seed)

        if is_sequence(dataset):
            data_loader_cls = DataLoader
        else:
            data_loader_cls = IterableDataLoader

        train_dataset = data_loader_cls(train_dataset, batch_size=len(train_dataset), drop_last=False, verbose=True)
        test_dataset = data_loader_cls(test_dataset, batch_size=len(test_dataset), drop_last=False, verbose=True)

        train_dataset, test_dataset = next(iter(train_dataset)), next(iter(test_dataset))

        return train_dataset, test_dataset

    def fit_preprocessors(self, dataset):
        self.dataset.preprocessors.fit(*dataset)

    def augment_dataset(self, dataset):
        augmented = self.dataset.data_augmentors.transform(dataset[0])
        return augmented, dataset[1]

    def transform_dataset(self, dataset):
        return self.dataset.preprocessors.transform(dataset[0]), dataset[1]

    def one_hot_encode(self, dataset):
        labels_info = self.dataset.dataset.labels_info
        ohe_labels = label_binarize(dataset[1], num_classes=labels_info.num_classes, multilabel=labels_info.multilabel)

        return dataset[0], ohe_labels

    def fit_model(self, dataset):
        return self.model.model.fit(*dataset, **self.model.training)

    def evaluate_model(self, dataset):
        return self.model.model.evaluate(*dataset)


class GeneratorTrainingRunner(TrainingRunner):
    """
    Runner for dataset loaded in batches with a generator.
    """

    def __init__(self, model: core.Model, dataset: core.Dataset):
        super().__init__(model, dataset)

        logger.info("Data will be loaded in batches with a generator. "
                    "Note that both preprocessors and model must support generators.")

    def load_dataset(self):
        return self._dataset_to_iterator(self.dataset.dataset)

    def _dataset_to_iterator(self, dataset):
        if is_sequence(self.dataset.dataset):
            data_loader_cls = DataLoader
        else:
            data_loader_cls = IterableDataLoader

        return data_loader_cls(dataset, batch_size=self.dataset.chunk_size, drop_last=False)

    def split_dataset(self, dataset):
        if is_sequence(self.dataset.dataset):
            split_func = split_sequence
        else:
            split_func = split_iterator

        splits = split_func(self.dataset.dataset, (self.dataset.samples.train, self.dataset.samples.test),
                            shuffle=self.dataset.shuffle, random_state=self.dataset.seed)

        return map(self._dataset_to_iterator, splits)

    def fit_preprocessors(self, dataset):
        self.dataset.preprocessors.fit_generator(dataset)

    def transform_dataset(self, dataset):
        def transform(batch):
            transformed = self.dataset.preprocessors.transform(batch[0])
            return transformed, batch[1]

        dataset.transform_func = chain(dataset.transform_func, transform)
        return dataset

    def augment_dataset(self, dataset):
        def augment(batch):
            augmented = self.dataset.data_augmentors.transform(batch[0])
            return augmented, batch[1]

        dataset.transform_func = chain(dataset.transform_func, augment)
        return dataset

    def one_hot_encode(self, dataset):
        def one_hot(batch):
            labels_info = self.dataset.dataset.labels_info
            ohe_labels = label_binarize(batch[1], num_classes=labels_info.num_classes,
                                        multilabel=labels_info.multilabel)
            return batch[0], ohe_labels

        dataset.transform_func = chain(dataset.transform_func, one_hot)
        return dataset

    def fit_model(self, dataset):
        return self.model.model.fit_generator(dataset, **self.model.training)

    def evaluate_model(self, dataset):
        return self.model.model.evaluate_generator(dataset)


class SparkTrainingRunner(TrainingRunner):
    """
    Runner for Spark datasets
    """

    def __init__(self, model: core.Model, dataset: core.Dataset):
        super().__init__(model, dataset)

        logger.info("Loading Spark dataset. Note that you need a Spark environment to run this training")

    def load_dataset(self):
        return self.dataset.dataset.load()

    def split_dataset(self, dataset):
        return split_spark_dataframe(dataset, (self.dataset.samples.train, self.dataset.samples.test),
                                     shuffle=self.dataset.shuffle, random_state=self.dataset.seed)

    def fit_preprocessors(self, dataset):
        self.dataset.preprocessors.fit(dataset)

    def augment_dataset(self, dataset):
        if self.dataset.data_augmentors:
            warnings.warn("Data augmentation not currently available for Spark datasets")
        return dataset

    def transform_dataset(self, dataset):
        return self.dataset.preprocessors.transform(dataset)

    def one_hot_encode(self, dataset):
        # TODO: one hot encode labels given number of classes
        raise ValueError("One-hot encoding not currently available for Spark datasets")

    def fit_model(self, dataset):
        return self.model.model.fit(dataset, **self.model.training)

    def evaluate_model(self, dataset):
        return self.model.model.evaluate(dataset)


# MARK: Trainer

class Trainer:
    """
    Base class to run trainings for supervised-learning datasets.

    Parameters
    -----------
    model: core.Model
        Gymnos model core object specifying model and training parameters
    dataset: core.Dataset
        Gymnos dataset core object specifying dataset, samples, preprocessors, etc ...
    tracking: core.Tracking, optional
        Gymnos tracking core object specifying trackers, options, etc ...
    """

    def __init__(self, model, dataset, tracking=None):
        if tracking is None:
            tracking = core.Tracking()

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
            model=core.Model(**model),
            dataset=core.Dataset(**dataset),
            tracking=core.Tracking(**tracking)
        )

    @staticmethod
    def load(path):
        """
        Load saved trainer.

        Parameters
        -----------
        path: str
            Path to load saved trainers.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            extract_zip(path, extract_dir=tempdir)

            with open(os.path.join(tempdir, "model.pkl"), "rb") as fp:
                model = core.Model(**dill.load(fp))

            with open(os.path.join(tempdir, "dataset.pkl"), "rb") as fp:
                dataset = core.Dataset(**dill.load(fp))

            with open(os.path.join(tempdir, "tracking.pkl"), "rb") as fp:
                tracking = core.Tracking(**dill.load(fp))

            model.model.restore(os.path.join(tempdir, "saved_model"))
            dataset.preprocessors.restore(os.path.join(tempdir, "saved_preprocessors"))

            with open(os.path.join(tempdir, "features_info.pkl"), "rb") as fp:
                features_info = dill.load(fp)

            with open(os.path.join(tempdir, "labels_info.pkl"), "rb") as fp:
                labels_info = dill.load(fp)

            # assign saved @property
            cls = type(dataset.dataset)
            cls = type(cls.__name__, (cls,), {})
            dataset.dataset.__class__ = cls
            setattr(cls, "features_info", features_info)
            setattr(cls, "labels_info", labels_info)

        return Trainer(model, dataset, tracking)

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

    def train(self, dl_manager=None, trackings_dir=None, callbacks=None, verbose=True):
        """
        Run trainer

        Parameters
        -----------
        dl_manager: services.DownloadManager, optional
            Download Manager
        trackings_dir: str, optional
            Directory to save tracking outputs. Defaults to current working directory
        callbacks: list of callbacks.Callback, optional
            Callbacks to run.
        verbose: bool
            Whether or not should print logs for each training step.

        Returns
        ------------
        dict
            - "elapsed": dict
                Elapsed times between steps:
                    * download_and_prepare_data
                    * load_data
                    * fit_preprocessors
                    * transform_data
                    * augment_data
                    * fit_model
                    * evaluate_model
            - "hardware_info": dict
                Dictionnary with hardware information (platform, gpus, cpu)
            - "metrics": list
                List with training and test metrics from model training and evaluation
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

        tracking_history = History()
        self.tracking.trackers.add(tracking_history)

        if verbose:
            callbacks.add(Logger())

        callbacks.on_train_begin()

        # MARK: Retrieve hardware info

        hardware_info = dict(platform=platform.platform())

        try:
            hardware_info["cpu"] = get_cpu_info()
        except Exception:
            logger.exception("Error retrieving CPU information")

        try:
            hardware_info["gpu"] = get_gpus_info()
        except Exception:
            logger.exception("Error retrieving GPU information")

        # MARK: Start tracking

        os.makedirs(trackings_dir, exist_ok=True)

        history_tracker = History()
        self.tracking.trackers.add(history_tracker)

        self.tracking.trackers.start(run_id=self.tracking.run_id, logdir=trackings_dir)

        if self.tracking.log_params:
            self.tracking.trackers.log_params(self.model.parameters)

        if self.tracking.log_training_params:
            self.tracking.trackers.log_params(self.model.training)

        self.tracking.trackers.log_tags(self.tracking.tags)

        # MARK: Download data

        callbacks.on_download_and_prepare_data_begin()

        self.dataset.dataset.download_and_prepare(dl_manager)

        callbacks.on_download_and_prepare_data_end()

        # MARK: Define runner to use

        if isinstance(self.dataset.dataset, SparkDataset):
            if isinstance(self.model.model, SparkModel):
                runner_cls = SparkTrainingRunner
            elif self.dataset.chunk_size is not None:
                runner_cls = GeneratorTrainingRunner
            else:
                raise ValueError("A SparkDataset is requested but the model is not a SparkModel or chunk_size is None")
        elif self.dataset.chunk_size is not None:
            runner_cls = GeneratorTrainingRunner
        else:
            runner_cls = InMemoryTrainingRunner

        runner = runner_cls(dataset=self.dataset, model=self.model)

        # MARK: Load dataset

        callbacks.on_load_data_begin()

        dataset = runner.load_dataset()

        train_dataset, test_dataset = runner.split_dataset(dataset)

        callbacks.on_load_data_end()

        # MARK: Data-augment dataset

        callbacks.on_augment_data_begin()

        train_dataset = runner.augment_dataset(train_dataset)

        callbacks.on_augment_data_end()

        # MARK: Fit preprocessors using train data

        callbacks.on_fit_preprocessors_begin()

        runner.fit_preprocessors(train_dataset)

        callbacks.on_fit_preprocessors_end()

        # MARK: Preprocess dataset

        callbacks.on_transform_data_begin()

        train_dataset = runner.transform_dataset(train_dataset)
        test_dataset = runner.transform_dataset(test_dataset)

        callbacks.on_transform_data_end()

        # MARK: One-hot encode labels if needed

        if self.dataset.one_hot:
            train_dataset = runner.one_hot_encode(train_dataset)
            test_dataset = runner.one_hot_encode(test_dataset)

        # MARK: Fit model using train data

        callbacks.on_fit_model_begin()

        train_metrics = runner.fit_model(train_dataset)

        callbacks.on_fit_model_end()

        if self.tracking.log_metrics:
            self.tracking.trackers.log_metrics(train_metrics)

        # MARK: Evaluate model using test data

        callbacks.on_evaluate_model_begin()

        test_metrics = runner.evaluate_model(test_dataset)

        callbacks.on_evaluate_model_end()

        if self.tracking.log_metrics:
            self.tracking.trackers.log_metrics(test_metrics, prefix="test_")

        return dict(
            elapsed=time_history.times,
            hardware_info=hardware_info,
            metrics=tracking_history.metrics
        )

    def predict(self, X):
        """
        Predict values.

        Parameters
        -----------
        X: np.ndarray, pd.Series or pyspark.sql.DataFrame
            Features

        Returns
        --------
        predictions: array-like or pyspark.sql.DataFrame (if X is a DataFrame)
            Predictions
        """
        X = self.dataset.preprocessors.transform(X)
        predictions = self.model.model.predict(X)
        return predictions

    def predict_proba(self, X):
        """
        Predict probabilities
        Parameters
        -----------
        X: np.ndarray, pd.Series or pyspark.sql.DataFrame
            Features
        Returns
        ---------
        predictions: array-like or pyspark.sql.DataFrame (if X is a DataFrame)
            Predictions
        """
        X = self.dataset.preprocessors.transform(X)
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

            model_store_dir = os.path.join(tempdir, "saved_model")
            os.makedirs(model_store_dir)
            self.model.model.save(model_store_dir)

            preprocessors_store_dir = os.path.join(tempdir, "saved_preprocessors")
            os.makedirs(preprocessors_store_dir)
            self.dataset.preprocessors.save(preprocessors_store_dir)

            zipdir(tempdir, path)
