#
#
#   Predictor
#
#

import os
import platform

import GPUtil
import cpuinfo

from .datasets import ClassificationDataset
from .logger import get_logger
from .utils.path import chdir
from .utils.timing import elapsed_time


class Predictor:
    """
    Entrypoint to run prediction given a model and a dataset.


    Parameters
    ----------
    trained_model_config_path: str
         Path with the directory where we load the traned model.
    scoring_table_path: str, optional
        Path with the directory and file where the dataset that will be scored.
    executions_dirname: str, optional
        Directory name where an execution for a dataset is saved.
    execution_format: str, optional
        Formatting for executions, it can contain named formatting options, which will be filled with
        the values of datetime, model_name, dataset_name and experiment_name.
    artifacts_dirname: str, optional
        Directory name where artifacts are saved (saved model, saved preprocessors, etc ...)
    cache_datasets_path: str, optional
        Directory to read and save HDF5 optimized datasets.
    """

    def __init__(self, trained_model_config_path="",
                 scoring_table_path="scorings", executions_dirname="executions",
                 execution_format="{datetime:%H-%M-%S--%d-%m-%Y}__{model_name}",
                 artifacts_dirname="artifacts", cache_datasets_path=None):
        self.trained_model_config_path = trained_model_config_path
        self.scoring_table_path = scoring_table_path
        self.executions_dirname = executions_dirname
        self.execution_format = execution_format
        self.artifacts_dirname = artifacts_dirname
        self.cache_datasets_path = cache_datasets_path

        self.logger = get_logger(prefix=self)

    def predict(self, model, dataset):
        """
        Run experiment generating outputs.

        Parameters
        ----------
        model: core.model.model
        dataset: core.dataset.Dataset

        Attributes
        ----------

        """
        execution_steps_elapsed = {}

        self.logger.info("Running prediction using trained model")

        # LOAD ARTIFACT PATH OF OLD EXPERIMENT

        trainings_dataset_execution_artifacts_path = os.path.join(self.trained_model_config_path,
                                                                  self.artifacts_dirname)

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
            self.logger.info("{}: {}".format(name, platform_details[key]))

        self.logger.info("Found {} GPUs".format(len(gpus_info)))

        # LOAD DATASET

        self.logger.info("Loading scoring table from path: {} ...".format(self.scoring_table_path))

        with elapsed_time() as elapsed:
            load_data_arguments = {}
            if self.cache_datasets_path is not None:
                load_data_arguments["hdf5_cache_path"] = os.path.join(self.cache_datasets_path, dataset.name + ".h5")

            if isinstance(dataset.dataset, ClassificationDataset):
                load_data_arguments["one_hot"] = dataset.one_hot

            if self.cache_datasets_path is None:
                dataset.dataset.read(self.scoring_table_path)
            else:
                scoring_table, y = dataset.dataset.load_data(**load_data_arguments)

        execution_steps_elapsed["load_data"] = elapsed.s
        self.logger.debug("Loading data took {:.2f}s".format(elapsed.s))

        # APPLY PREPROCESSORS

        self.logger.info("Restoring preprocessors")

        with elapsed_time() as elapsed:
            dataset.preprocessor_pipeline.restore(
                os.path.join(trainings_dataset_execution_artifacts_path, "pipeline.pkl"))

        execution_steps_elapsed["restore_preprocessors"] = elapsed.s
        self.logger.debug("Restoring preprocessors to train data took {:.2f}s".format(elapsed.s))

        with elapsed_time() as elapsed:
            preprocessed_scoring_table = dataset.preprocessor_pipeline.transform(scoring_table, data_desc="X_train")

        execution_steps_elapsed["transform_preprocessors"] = elapsed.s
        self.logger.debug("Preprocessing data took {:.2f}s".format(elapsed.s))

        # RESTORE MODEL

        self.logger.info("Restore model")
        with elapsed_time() as elapsed, chdir(trainings_dataset_execution_artifacts_path):
            model.model.restore(trainings_dataset_execution_artifacts_path)

        execution_steps_elapsed["Restore model"] = elapsed.s
        self.logger.debug("Restoring prediction took {:.2f}s".format(elapsed.s))

        # PREDICT

        self.logger.info("Apply model")
        with elapsed_time() as elapsed, chdir(trainings_dataset_execution_artifacts_path):
            prediction = model.model.predict(preprocessed_scoring_table)

        execution_steps_elapsed["Apply model"] = elapsed.s
        self.logger.debug("Applying model took {:.2f}s".format(elapsed.s))

        return prediction
