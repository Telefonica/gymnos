#
#
#   Data Usage Linear Regression
#
#

import numpy as np
from sklearn.linear_model import LinearRegression

from .model import ScikitLearnModel
from ..utils.temporal_series_utils import mad_mean_error, nrmsd_error_norm, residual_analysis, rmse_train


class DataUsageLinearRegression(ScikitLearnModel):

    def __init__(self, input_shape, **hyperparameters):
        super().__init__(input_shape, sklearn_model=LinearRegression())

        self.n_preds = hyperparameters.get("n_preds", 3)

    def evaluate(self, X, y, batch_size=32, verbose=0):
        """
        Evaluates with the following metrics.

        mean_error: (float) mean Absolute Deviation/Mean ratio
        normal_error: (float) root mean square deviation
        residual_error: (bool) True/False according to independence test result
        emc_error: (float) quadratic mean error for execution date

        """
        y_pred = self.predict(X, batch_size=batch_size, verbose=verbose)

        y_pred = [i.tolist()[0] for i in y_pred]
        y_pred = y_pred[-self.n_preds:]

        y = y.flatten().tolist()

        return {
            "mean_error": mad_mean_error(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "normal_error": nrmsd_error_norm(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "residual_error": residual_analysis(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "emc_error": rmse_train(y[-self.n_preds:], y_pred[-self.n_preds:])
        }
