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

    def predict(self, X):
        """
        Run experiment generating outputs.

        Parameters
        ----------
        X: dataset used for prediction

        """
        # RETRIEVE PLATFORM DETAILS

        for name, key in zip(("Python version", "Platform"), ("python_version", "platform")):
            self.logger.info("{}: {}".format(name, platform_details(key)))

        self.logger.info("Found {} GPUs".format(len(platform_details("gpu"))))

        # APPLY PREPROCESSORS

        self.logger.info("Apply preprocessors")

        with elapsed_time() as elapsed:
            X = self.dataset.preprocessor_pipeline.transform(X)

        self.logger.debug("Applying preprocessors took {:.2f}s".format(elapsed.s))

        # PREDICT

        self.logger.info("Apply model")
        with elapsed_time() as elapsed:
            prediction = self.model.model.predict(X)

        self.logger.debug("Applying model took {:.2f}s".format(elapsed.s))
        return prediction
