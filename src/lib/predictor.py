#
#
#   Predictor
#
#

import logging

from .utils.platform_details import platform_details
from .utils.timing import ElapsedTimeCalculator

logger = logging.getLogger(__name__)


class Predictor:
    """
    Entrypoint to run prediction given a model and a dataset.

    """

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def predict(self, X):
        """
        Run experiment generating outputs.

        Parameters
        ----------
        X: dataset used for prediction

        """
        elapsed_time_calc = ElapsedTimeCalculator()

        # RETRIEVE PLATFORM DETAILS

        for name, key in zip(("Python version", "Platform"), ("python_version", "platform")):
            logger.info("{}: {}".format(name, platform_details(key)))

        logger.info("Found {} GPUs".format(len(platform_details("gpu"))))

        # APPLY PREPROCESSORS

        logger.info("Apply preprocessors")

        with elapsed_time_calc("preprocessors_transform") as elapsed:
            X = self.dataset.preprocessors.transform(X)

        logger.debug("Applying preprocessors took {:.2f}s".format(elapsed.s))

        # PREDICT

        logger.info("Predicting with model")
        with elapsed_time_calc("predict_model") as elapsed:
            prediction = self.model.model.predict(X)

        logger.debug("Predicting with model took {:.2f}s".format(elapsed.s))
        return prediction
