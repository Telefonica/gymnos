#
#
#   Predictor
#
#
from .logger import get_logger
from .utils.platform_details import platform_details
from .utils.timing import elapsed_time


class Predictor:
    """
    Entrypoint to run prediction given a model and a dataset.

    """

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.logger = get_logger(prefix=self)

    def predict(self, values):
        """
        Run experiment generating outputs.

        Parameters
        ----------
        values: dataset used for prediction

        """

        # RETRIEVE PLATFORM DETAILS

        execution_steps_elapsed = {}

        for name, key in zip(("Python version", "Platform"), ("python_version", "platform")):
            self.logger.info("{}: {}".format(name, platform_details(key)))

        self.logger.info("Found {} GPUs".format(len(platform_details("gpu"))))

        # APPLY PREPROCESSORS

        self.logger.info("Apply preprocessors")

        with elapsed_time() as elapsed:
            values = self.dataset.preprocessor_pipeline.transform(values)

        execution_steps_elapsed["transform_preprocessors"] = elapsed.s
        self.logger.debug("Preprocessing data took {:.2f}s".format(elapsed.s))

        # PREDICT

        self.logger.info("Apply model")
        with elapsed_time() as elapsed:
            prediction = self.model.model.predict(values)

        execution_steps_elapsed["Apply model"] = elapsed.s
        self.logger.debug("Applying model took {:.2f}s".format(elapsed.s))

        return prediction
