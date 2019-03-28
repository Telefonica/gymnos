#
#
#   Trainer
#
#

from pprint import pprint

from .logger import get_logger
from .utils.iterator_utils import count
from .utils.ml_utils import train_val_test_split


class Trainer:

    def __init__(self, experiment, model, dataset, training, session=None, tracking=None):
        self.experiment = experiment
        self.model = model
        self.dataset = dataset
        self.training = training
        self.session = session
        self.tracking = tracking

        self.logger = get_logger(prefix=self)

    def run(self, seed=0):
        self.logger.info("Running experiment: {} ...".format(self.experiment.creation_date))

        self.logger.info("Loading dataset: {} ...".format(self.dataset.name))
        X, y = self.dataset.dataset.load_data()

        self.logger.info("Splitting dataset -> Fit: {} | Test: {} | Val: {} ...".format(
                         self.training.samples.fit,  self.training.samples.test, self.training.samples.val))
        (X_train, X_val, X_test), (y_train, y_val, y_test) = train_val_test_split(X, y,
                                                                                  train_size=self.training.samples.fit,
                                                                                  val_size=self.training.samples.val,
                                                                                  test_size=self.training.samples.test,
                                                                                  seed=seed)
        # APPLY PREPROCESSORS

        self.logger.info("Applying {} preprocessors ...".format(len(self.dataset.preprocessor_stack)))

        X_train = self.dataset.preprocessor_stack.transform(X_train)
        X_test = self.dataset.preprocessor_stack.transform(X_test)
        X_val = self.dataset.preprocessor_stack.transform(X_val)

        # APPLY TRANSFORMERS

        self.logger.info("Applying {} transformers ...".format(len(self.dataset.transformer_stack)))

        self.dataset.transformer_stack.fit(X_train, y_train)

        X_train = self.dataset.transformer_stack.transform(X_train)
        X_test = self.dataset.transformer_stack.transform(X_test)
        X_val = self.dataset.transformer_stack.transform(X_val)

        self.logger.info("Fitting model with {} samples ...".format(count(X_train)))

        val_data = None

        if self.training.samples.val > 0:
            val_data = [X_val, y_val]

        train_results = self.model.model.fit(X_train, y_train, batch_size=self.training.batch_size,
                                             epochs=self.training.epochs, val_data=val_data,
                                             callbacks=self.training.callbacks + self.tracking.get_keras_callbacks())
        pprint(train_results)

        # EVALUATE MODEL IF TEST SAMPLES

        if self.training.samples.test > 0:

            self.logger.info("Evaluating model with {} samples".format(count(X_test)))

            test_results = self.model.model.evaluate(X_test, y_test)

            pprint(test_results)

            self.logger.info("Logging metrics to {} trackers".format(len(self.tracking.tracker_list)))
            self.tracking.tracker_list.log_metrics(test_results, prefix="test_")
