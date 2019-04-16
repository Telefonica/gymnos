#
#
#   Trainer
#
#

import os
import platform
import numpy as np
import tensorflow as tf
from datetime import datetime
from pprint import pprint

from . import trackers
from .logger import get_logger
from .utils.io_utils import save_to_json
from .utils.iterator_utils import count
from .utils.ml_utils import train_val_test_split
from .utils.timing import elapsed_time

TRAINING_MODEL_NAME = "model"
TRAININGS_FOLDERNAME = "trainings"
TRAINING_METRICS_FILENAME = "metrics.json"
TRAINING_CALLBACKS_FOLDERNAME = "callbacks"
TRAINING_TRACKINGS_FOLDERNAME = "trackings"
TRAINING_EXECUTIONS_FOLDERNAME = "executions"
TRAINING_EXECUTION_ID_STRFTIME = "%H-%M-%S__%d-%m-%Y"


class Trainer:
    """
    Entrypoint to run experiment given an experiment, a model, a dataset, a training and optionally a
    session and a tracking.
    The run method will create a directory with the following structure:
    TRAININGS_FOLDERNAME/
    └── dataset.name/
        ├── TRAINING_EXECUTIONS_FOLDERNAME/
        │   └── TRAINING_EXECUTION_ID_STRFTIME/  # each execution will be named with the datetime the experiment is run
        │       ├── TRAINING_CALLBACKS_FOLDERNAME/  # artifacts generated by each callback
        │       │   └── training.callbacks[i].name/
        │       ├── TRAINING_METRICS_FILENAME  # filename to keep metrics generated and elapsed times
        │       ├── TRAINING_MODEL_NAME + <extension>  # model weights/parameters
        └── TRAINING_TRACKINGS_FOLDERNAME/  # artifacts generated by each tracker
            └── tracking.trackers[i].name/
    """

    def __init__(self, experiment, model, dataset, training, session=None, tracking=None):
        self.experiment = experiment
        self.model = model
        self.dataset = dataset
        self.training = training
        self.session = session
        self.tracking = tracking

        self.logger = get_logger(prefix=self)

    def run(self, seed=0):
        execution_steps_elapsed = {}
        execution_id = datetime.now().strftime(TRAINING_EXECUTION_ID_STRFTIME)

        self.logger.info("Running experiment: {} ...".format(execution_id))

        # CREATE DIRECTORIES TO STORE TRAININGS EXECUTIONS

        trainings_dataset_path = os.path.join(TRAININGS_FOLDERNAME, self.dataset.name)
        trainings_dataset_trackings_path = os.path.join(trainings_dataset_path, TRAINING_TRACKINGS_FOLDERNAME)
        trainings_dataset_execution_path = os.path.join(trainings_dataset_path, TRAINING_EXECUTIONS_FOLDERNAME,
                                                        execution_id)
        os.makedirs(trainings_dataset_execution_path, exist_ok=True)
        self.logger.info("Creating directory to save training results ({})".format(trainings_dataset_execution_path))

        # CONFIGURE TRACKERS AND CALLBACKS TO STORE ARTIFACTS TO CURRENT EXECUTION DIRECTORY

        self.tracking.configure_trackers(logdir=trainings_dataset_trackings_path, run_name=execution_id)
        self.training.configure_callbacks(base_dir=os.path.join(trainings_dataset_execution_path,
                                                                TRAINING_CALLBACKS_FOLDERNAME))

        # RETRIEVE PLATFORM DETAILS

        platform_details = {
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "system": platform.system(),
            "node": platform.node(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "is_gpu_available": bool(tf.test.gpu_device_name())
        }

        for name, key in zip(("Python version", "Platform"), ("python_version", "platform")):
            self.logger.debug("{}: {}".format(name, platform_details[key]))

        # LOG HYPERPARAMETERS
        self.tracking.trackers.log_params(self.model.hyperparameters)

        # LOAD DATASET AND SPLIT IT FOR CROSS VALIDATION

        self.logger.info("Loading dataset: {} ...".format(self.dataset.name))

        with elapsed_time() as elapsed:
            X, y = self.dataset.dataset.load_data()

        execution_steps_elapsed["load_data"] = elapsed.s
        self.logger.debug("Loading data took {:.2f}s".format(elapsed.s))

        self.logger.info("Splitting dataset -> Fit: {} | Test: {} | Val: {} ...".format(
            self.training.samples.fit, self.training.samples.test, self.training.samples.val))
        (X_train, X_val, X_test), (y_train, y_val, y_test) = train_val_test_split(X, y,
                                                                                  train_size=self.training.samples.fit,
                                                                                  val_size=self.training.samples.val,
                                                                                  test_size=self.training.samples.test,
                                                                                  seed=seed,
                                                                                  shuffle=self.training.shuffle)
        # APPLY PREPROCESSORS

        self.logger.info("Applying {} preprocessors ({})".format(len(self.dataset.preprocessor_pipeline),
                                                                 str(self.dataset.preprocessor_pipeline)))

        with elapsed_time() as elapsed:
            self.dataset.preprocessor_pipeline.fit(X_train, y_train)

        execution_steps_elapsed["fit_preprocessors"] = elapsed.s
        self.logger.debug("Fitting preprocessors to train data took {:.2f}s".format(elapsed.s))

        with elapsed_time() as elapsed:
            X_train = self.dataset.preprocessor_pipeline.transform(X_train, data_desc="X_train")
            X_test = self.dataset.preprocessor_pipeline.transform(X_test, data_desc="X_test")
            X_val = self.dataset.preprocessor_pipeline.transform(X_val, data_desc="X_val")

        execution_steps_elapsed["transform_preprocessors"] = elapsed.s
        self.logger.debug("Preprocessing data took {:.2f}s".format(elapsed.s))

        # DEFINE PLACEHOLDER TO KEEP TRAIN, TEST, VAL METRICS

        history_tracker = trackers.History()
        self.tracking.trackers.add(history_tracker)

        # FIT MODEL

        self.logger.info("Fitting model with {} samples ...".format(count(X_train)))

        val_data = None

        if self.training.samples.val > 0:
            val_data = [X_val, y_val]

        with elapsed_time() as elapsed:
            train_metrics = self.model.model.fit(X_train, y_train, batch_size=self.training.batch_size,
                                                 epochs=self.training.epochs, val_data=val_data,
                                                 callbacks=self.training.callbacks)
        execution_steps_elapsed["fit_model"] = elapsed.s
        self.logger.debug("Fitting model took {:.2f}s".format(elapsed.s))

        pprint(train_metrics)

        for metric_name, metric_value in train_metrics.items():
            self.logger.info("Results for {}: Min: {:.2f} | Max: {:.2f} | Mean: {:.2f}".format(metric_name,
                                                                                               np.min(metric_value),
                                                                                               np.max(metric_value),
                                                                                               np.mean(metric_value)))
        self.logger.info("Logging train metrics to trackers".format(len(self.tracking.trackers)))
        self.tracking.trackers.log_metrics(train_metrics)

        # EVALUATE MODEL IF TEST SAMPLES EXIST

        if self.training.samples.test > 0:
            self.logger.info("Evaluating model with {} samples".format(count(X_test)))

            with elapsed_time() as elapsed:
                test_metrics = self.model.model.evaluate(X_test, y_test)

            execution_steps_elapsed["evaluate_model"] = elapsed.s
            self.logger.debug("Evaluating model took {:.2f}s".format(elapsed.s))

            pprint(test_metrics)

            self.logger.info("Logging test metrics to trackers".format(len(self.tracking.trackers)))
            self.tracking.trackers.log_metrics(test_metrics, prefix="test_")

        # Log additional params

        self.tracking.trackers.log_params(self.tracking.params)

        # SAVE MODEL

        self.logger.info("Saving model")
        self.model.model.save(trainings_dataset_execution_path, name=TRAINING_MODEL_NAME)

        metrics = dict(
            elapsed=execution_steps_elapsed,
            metrics=history_tracker.metrics,
            platform=platform_details
        )
        save_to_json(os.path.join(trainings_dataset_execution_path, TRAINING_METRICS_FILENAME), metrics)

        self.logger.info("Metrics, platform information and elapsed times saved to {} file".format(
                         TRAINING_METRICS_FILENAME))

        self.tracking.trackers.end()

        return trainings_dataset_execution_path
