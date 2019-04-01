#
#
#   Data Usage Linear Regression
#
#

import numpy as np
from sklearn.linear_model import LinearRegression

from .model import ScikitLearnModel
from .. import trackers
from ..utils.temporal_series_utils import mad_mean_error, nrmsd_error_norm, residual_analysis, rmse_train


class DataUsageLinearRegression(ScikitLearnModel):

    def __init__(self, input_shape, **hyperparameters):
        super().__init__(input_shape, sklearn_model=LinearRegression())

        self.n_preds = hyperparameters.get("n_preds", 3)

    def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        history = trackers.History()
        history.log_metrics({})
        return history.metrics

    def predict(self, X, batch_size=32, verbose=0):
        """
        Predict the next values of a series by applying Linear Regression model

        """
        X = list(X)

        # Erase zeros on the left
        consumption_zero = [i for i in X if i > 0.0]

        # LR need at least two days with positive data
        if len(consumption_zero) >= 2:

            consumption = np.cumsum(X).tolist()
            consumption_zero_acum = [i for i in consumption if i > 0.0]
            consumption_zero_acum = [float(i) for i in consumption_zero_acum]

            # Training
            self.model.fit(np.array(range(len(consumption_zero_acum))).reshape(-1, 1),
                           np.array(consumption_zero_acum).reshape(-1, 1))

            # Prediction
            result = self.model.predict(np.array(range(len(X) + self.n_preds)).reshape(-1, 1))
            future_predictions = [i.tolist()[0] for i in result]
            future_predictions = future_predictions[-self.n_preds:]
        else:
            future_predictions = [0] * self.n_preds

        return future_predictions

    def evaluate(self, X, y, batch_size=32, verbose=0):
        """
        Evaluates with the following metrics.

        mean_error: (float) mean Absolute Deviation/Mean ratio
        normal_error: (float) root mean square deviation
        residual_error: (bool) True/False according to independence test result
        emc_error: (float) quadratic mean error for execution date

        """
        y_pred = self.predict(X, batch_size=batch_size, verbose=verbose)

        y = list(y)
        print("y" + str(y))
        print("y_pred" + str(y_pred))

        if (y[-self.n_preds:] and y_pred[-self.n_preds:] and len(y[-self.n_preds:]) == len(y_pred[-self.n_preds:])
                and len(y_pred[-self.n_preds:]) > 0):
            mean_error = mad_mean_error(y[-self.n_preds:], y_pred[-self.n_preds:])
            normal_error = nrmsd_error_norm(y[-self.n_preds:], y_pred[-self.n_preds:])
            residual_error = residual_analysis(y[-self.n_preds:], y_pred[-self.n_preds:])
            emc_error = rmse_train(y[-self.n_preds:], y_pred[-self.n_preds:])

        else:
            mean_error = 0.0
            normal_error = 0.0
            residual_error = True
            emc_error = 0.0
        return {
            "mean_error": mean_error,
            "normal_error": normal_error,
            "residual_error": residual_error,
            "emc_error": emc_error}

    def restore(self, file_path):
        pass

    def save(self, directory, name="model"):
        pass
