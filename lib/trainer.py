#
#
#   Trainer
#
#

from pprint import pprint

from .logger import logger
from .utils.ml_utils import train_val_test_split


class Trainer:

    def __init__(self, experiment, model, dataset, training, session=None, tracking=None):
        self.experiment = experiment
        self.model = model
        self.dataset = dataset
        self.training = training
        self.session = session
        self.tracking = tracking


    def run(self, seed=0):
        logger.info("Running experiment: {} ...".format(self.experiment.id))

        logger.info("Loading dataset: {} ...".format(self.dataset.id))
        X, y = self.dataset.dataset.load_data()

        logger.info("Splitting dataset for cross-validation ->\tFit: {}\tTest: {}\tVal: {} ...".format(
                    self.training.samples.fit,  self.training.samples.test, self.training.samples.val))
        (X_train, X_val, X_test), (y_train, y_val, y_test) = train_val_test_split(X, y,
                                                                                  train_size=self.training.samples.fit,
                                                                                  val_size=self.training.samples.val,
                                                                                  test_size=self.training.samples.test,
                                                                                  seed=seed)
        # APPLY PREPROCESSORS

        logger.info("Applying {} preprocessors ...".format(len(self.dataset.preprocessor_stack)))

        X_train = self.dataset.preprocessor_stack.transform(X_train)
        X_test = self.dataset.preprocessor_stack.transform(X_test)
        X_val = self.dataset.preprocessor_stack.transform(X_val)

        # APPLY TRANSFORMERS

        logger.info("Applying {} transformers ...".format(len(self.dataset.transformer_stack)))

        self.dataset.transformer_stack.fit(X_train, y_train)

        X_train = self.dataset.transformer_stack.transform(X_train, y_train)
        X_test = self.dataset.transformer_stack.transform(X_test, y_test)
        X_val = self.dataset.transformer_stack.transform(X_val, y_val)

        # TRAIN MODEL
        if self.model.compilation is not None:
            logger.info("Compiling model ...")

            self.model.model.compile(loss=self.model.compilation.loss, optimizer=self.model.compilation.optimizer,
                                     metrics=self.model.compilation.metrics)

        logger.info("Fitting model with {} samples ...".format(len(X_train)))

        train_results = self.model.model.fit(X_train, y_train, batch_size=self.training.batch_size,
                                             epochs=self.training.epochs, val_data=[X_val, y_val],
                                             callbacks=self.training.callbacks + self.tracking.get_keras_callbacks())

        pprint(train_results)

        # EVALUATE MODEL

        logger.info("Evaluating model ...")

        test_results = self.model.model.evaluate(X_test, y_test)

        pprint(test_results)

        self.tracking.tracker_list.log_metrics(test_results, prefix="test_")
