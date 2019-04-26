#
#
#   Data Usage Linear Regression
#
#

from sklearn.linear_model import LinearRegression

from .model import Model
from .mixins import SklearnMixin
from ..utils.temporal_series_utils import mad_mean_error, nrmsd_error_norm, residual_analysis, rmse_train


class DataUsageLinearRegression(Model, SklearnMixin):
    """
    Linear Regression developed to solve Data Usage time series regression task.

    Parameters
    ----------
    n_preds: int, optional
        Number of days to predict.
    """

    def __init__(self, n_preds=3):
        self.model = LinearRegression()
        self.n_preds = n_preds

    def evaluate(self, X, y):
        y_pred = self.predict(X)

        y_pred = [i.tolist()[0] for i in y_pred]
        y_pred = y_pred[-self.n_preds:]

        y = y.flatten().tolist()

        return {
            "mean_error": mad_mean_error(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "normal_error": nrmsd_error_norm(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "residual_error": residual_analysis(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "emc_error": rmse_train(y[-self.n_preds:], y_pred[-self.n_preds:])
        }
