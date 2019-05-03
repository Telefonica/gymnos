#
#
#   Trainer
#
#

import os
import GPUtil
import cpuinfo
import platform
import numpy as np

from datetime import datetime
from sklearn.model_selection import train_test_split

from . import trackers
from .logger import get_logger
from .utils.path import chdir
from .utils.io_utils import save_to_json
from .utils.timing import elapsed_time
from .datasets import ClassificationDataset


class Trainer:
    """
    Entrypoint to run experiment given an experiment, a model, a dataset, a training and a tracking.
    The run method will create a directory with the following structure:
    trainings_path/
    └── {{dataset.name}}/
        ├── executions_dirname/
        │   └── {{datetime ~ execution_format}}/  # datetime when run is called
        │       ├── metrics.json  # metrics, HW details and elapsed times
        │       └── artifacts
        │          └── ... # model weights/parameters, trainings artifacts (e.g callbacks),
        │                     preprocessors pipeline, etc ...
        └── trackings_dirname/  # artifacts generated by each tracker
            └── {{tracking.trackers[i].name}}/

    Parameters
    ----------
    trainings_path: str, optional
        Path with the directory where the experiments are run.
    executions_dirname: str, optional
        Directory name where an execution for a dataset is saved.
    trackings_dirname: str, optional
        Directory name where an execution for a dataset is tracked.
    execution_format: str, optional
        Formatting for executions, it can contain named formatting options, which will be filled with
        the values of datetime, model_name, dataset_name and experiment_name.
    artifacts_dirname: str, optional
        Directory name where artifacts are saved (saved model, saved preprocessors, etc ...)
    cache_datasets_path: str, optional
        Directory to read and save HDF5 optimized datasets.
    """

    def __init__(self, trainings_path="trainings", executions_dirname="executions",
                 trackings_dirname="trackings", execution_format="{datetime:%H-%M-%S--%d-%m-%Y}__{model_name}",
                 artifacts_dirname="artifacts", cache_datasets_path=None):
        self.trainings_path = trainings_path
        self.executions_dirname = executions_dirname
        self.trackings_dirname = trackings_dirname
        self.execution_format = execution_format
        self.artifacts_dirname = artifacts_dirname
        self.cache_datasets_path = cache_datasets_path

        self.logger = get_logger(prefix=self)


    def train(self, experiment, model, dataset, training, tracking):
        """
        Run experiment generating outputs.

        Parameters
        ----------
        experiment: core.experiment.Experiment
        model: core.model.model
        dataset: core.dataset.Dataset
        training: core.training.Training
        tracking: core.tracking.Tracking

        Returns
        -------
        trainings_dataset_execution_path: str
            Path of execution.
        """

        execution_steps_elapsed = {}
        execution_id = self.execution_format.format(datetime=datetime.now(), model_name=model.name,
                                                    dataset_name=dataset.name, experiment_name=experiment.name)

        self.logger.info("Running experiment: {} ...".format(execution_id))

        # CREATE DIRECTORIES TO STORE TRAININGS EXECUTIONS

        trainings_dataset_path = os.path.join(self.trainings_path, dataset.name)
        trainings_dataset_trackings_path = os.path.join(trainings_dataset_path, self.trackings_dirname)
        trainings_dataset_execution_path = os.path.join(trainings_dataset_path, self.executions_dirname, execution_id)
        trainings_dataset_execution_artifacts_path = os.path.join(trainings_dataset_execution_path,
                                                                  self.artifacts_dirname)

        os.makedirs(trainings_dataset_execution_path)
        os.makedirs(trainings_dataset_execution_artifacts_path)
        os.makedirs(trainings_dataset_trackings_path, exist_ok=True)

        self.logger.info("The execution will be saved in the following directory: {}".format(
                         trainings_dataset_execution_path))
        self.logger.info("Tracking information will be saved in the following directory: {}".format(
                         trainings_dataset_trackings_path))

        # RETRIEVE PLATFORM DETAILS

        cpu_info = cpuinfo.get_cpu_info()

        gpus_info = []
        for gpu in GPUtil.getGPUs():
            gpus_info.append({
                "name": gpu.name,
                "memory": gpu.memoryTotal
            })

        platform_details = {
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
            self.logger.debug("{}: {}".format(name, platform_details[key]))

        self.logger.debug("Found {} GPUs".format(len(gpus_info)))

        # DEFINE TRACKER TO STORE METRICS AND SAVE THEM TO JSON LATER

        history_tracker = trackers.History()
        tracking.trackers.add(history_tracker)

        # START TRACKING

        tracking.trackers.start(run_name=execution_id, logdir=trainings_dataset_trackings_path)

        # LOG TRACKING AND MODEL PARAMETERS

        tracking.trackers.log_params(tracking.params)
        tracking.trackers.log_params(model.parameters)

        # LOAD DATASET

        self.logger.info("Loading dataset: {} ...".format(dataset.name))

        with elapsed_time() as elapsed:
            load_data_arguments = {}
            if self.cache_datasets_path is not None:
                load_data_arguments["hdf5_cache_path"] = os.path.join(self.cache_datasets_path, dataset.name + ".h5")
            if isinstance(dataset.dataset, ClassificationDataset):
                load_data_arguments["one_hot"] = dataset.one_hot

            X, y = dataset.dataset.load_data(**load_data_arguments)

        execution_steps_elapsed["load_data"] = elapsed.s
        self.logger.debug("Loading data took {:.2f}s".format(elapsed.s))

        # SPLIT DATASET INTO TRAIN AND TEST

        self.logger.info("Splitting dataset -> Train: {:.2f} | Test: {:.2f}".format(dataset.samples.train,
                                                                                    dataset.samples.test))
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=dataset.samples.train,
                                                            test_size=dataset.samples.test,
                                                            shuffle=dataset.shuffle,
                                                            random_state=dataset.seed)
        # APPLY PREPROCESSORS

        self.logger.info("Applying {} preprocessors ({})".format(len(dataset.preprocessor_pipeline),
                                                                 str(dataset.preprocessor_pipeline)))

        with elapsed_time() as elapsed:
            dataset.preprocessor_pipeline.fit(X_train, y_train)

        execution_steps_elapsed["fit_preprocessors"] = elapsed.s
        self.logger.debug("Fitting preprocessors to train data took {:.2f}s".format(elapsed.s))

        with elapsed_time() as elapsed:
            X_train = dataset.preprocessor_pipeline.transform(X_train, data_desc="X_train")
            X_test = dataset.preprocessor_pipeline.transform(X_test, data_desc="X_test")

        execution_steps_elapsed["transform_preprocessors"] = elapsed.s
        self.logger.debug("Preprocessing data took {:.2f}s".format(elapsed.s))

        # FIT MODEL

        self.logger.info("Fitting model with {} samples ...".format(len(X_train)))

        # Measure time and temporary change directory in case model wants to save some artifact while fitting
        with elapsed_time() as elapsed, chdir(trainings_dataset_execution_artifacts_path):
            train_metrics = model.model.fit(X_train, y_train, **training.parameters)

        execution_steps_elapsed["fit_model"] = elapsed.s
        self.logger.debug("Fitting model took {:.2f}s".format(elapsed.s))

        for metric_name, metric_value in train_metrics.items():
            self.logger.info("Results for {}: Min: {:.2f} | Max: {:.2f} | Mean: {:.2f}".format(metric_name,
                                                                                               np.min(metric_value),
                                                                                               np.max(metric_value),
                                                                                               np.mean(metric_value)))
        self.logger.info("Logging train metrics to trackers".format(len(tracking.trackers)))
        tracking.trackers.log_metrics(train_metrics)

        # EVALUATE MODEL IF TEST SAMPLES EXIST

        self.logger.info("Evaluating model with {} samples".format(len(X_test)))

        with elapsed_time() as elapsed:
            test_metrics = model.model.evaluate(X_test, y_test)

        for metric_name, metric_value in test_metrics.items():
            self.logger.info("Test results for {}: {:.2f}".format(metric_name, np.mean(metric_value)))

        execution_steps_elapsed["evaluate_model"] = elapsed.s
        self.logger.debug("Evaluating model took {:.2f}s".format(elapsed.s))

        self.logger.info("Logging test metrics to trackers".format(len(tracking.trackers)))
        tracking.trackers.log_metrics(test_metrics, prefix="test_")

        # SAVE MODEL

        self.logger.info("Saving model")
        model.model.save(trainings_dataset_execution_artifacts_path)

        # SAVE PIPELINE

        self.logger.info("Saving pipeline")
        dataset.preprocessor_pipeline.save(os.path.join(trainings_dataset_execution_artifacts_path, "pipeline.pkl"))

        # SAVE METRICS

        self.logger.info("Saving metrics to JSON file")
        metrics = dict(
            elapsed=execution_steps_elapsed,
            metrics=history_tracker.metrics,
            platform=platform_details
        )
        metrics_path = os.path.join(trainings_dataset_execution_path, "metrics.json")
        save_to_json(metrics_path, metrics)

        self.logger.info("Metrics, platform information and elapsed times saved to {} file".format(metrics_path))

        tracking.trackers.end()

        return trainings_dataset_execution_path
