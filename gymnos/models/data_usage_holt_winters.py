#
#
#   Data usage Holt-Winters
#
#

import numpy as np

from .model import Model
from ..utils.lazy_imports import lazy_imports as lazy
from .utils.temporal_series_utils import estimated_window, initial_trend, initial_seasonal_components, rmse_holt_winters
from .utils.temporal_series_utils import mad_mean_error, nrmsd_error_norm, residual_analysis, rmse_train


class DataUsageHoltWinters(Model):
    """
    Task: **Regression**

    Holt Winters algorithm developed to solve Data Usage time series regression
    task (:class:`gymnos.datasets.data_usage_test.DataUsageTest`).

    Can I use?
        - Generators: ❌
        - Probability predictions: ❌
        - Distributed datasets: ❌

    Parameters
    ----------
    n_preds: int, optional
        Number of days to predict
    min_historic: int, optional
        TODO
    flag_optimize_hiperparams: bool, optional
        TODO
    slen: int, optional
        TODO
    alpha: float, optional
        TODO
    beta: float, optional
        TODO
    gamma: float, optional
        TODO
    """

    def __init__(self, n_preds=17, min_historic=5, flag_optimize_hiperparams=True, slen=1, alpha=0.716,
                 beta=0.029, gamma=0.993):
        self.min_historic = min_historic
        self.flag_optimize_hiperparams = flag_optimize_hiperparams
        self.n_preds = n_preds
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, X, y):
        """
        Holt Winters doesn't need to fit model to training data.
        """
        return {}

    def fit_generator(self, generator):
        return {}

    def predict(self, X):
        if self.flag_optimize_hiperparams:
            slen = estimated_window(X, 2)

            if (slen != -1) and (len(X) > int(self.min_historic)):
                initial_values = np.array([0.3, 0.1, 0.1])
                boundaries = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

                scipy = __import__("{}.optimize".format(lazy.scipy.__name__))

                parameters = scipy.optimize.fmin_l_bfgs_b(rmse_holt_winters,
                                                          x0=initial_values,
                                                          args=(X[:], 'additive', self.slen),
                                                          bounds=boundaries,
                                                          approx_grad=True)

                self.alpha, self.beta, self.gamma = parameters[0]
                self.slen = slen

        # Holt Winters model
        result = []
        smooth = X[0]
        trend = initial_trend(X, self.slen)
        seasonals = initial_seasonal_components(X, self.slen)

        result.append(X[0])

        for i in range(len(X) + self.n_preds - 1):

            # We are forecasting
            if i >= (len(X) - 1):
                m = i - len(X) + 1
                prediction = (smooth + m * trend) + seasonals[i % self.slen]

                if prediction < 0.0:
                    result.append(0.0)
                else:
                    result.append(prediction)

            # To know how much days user can continue with his normal consumption after his billing_end_date:
            else:
                last_smooth = smooth
                smooth = self.alpha * (X[i] - seasonals[i % self.slen]) + (1 - self.alpha) * (smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = (self.gamma * (X[i] - smooth) + (1 - self.gamma) * seasonals[i % self.slen])
                result.append(smooth + trend + seasonals[i % self.slen])

        future_predictions = result[-self.n_preds:]

        return future_predictions

    def evaluate(self, X, y):
        y[1:] -= y[:-1].copy()
        y = y.flatten().tolist()

        y_pred = self.predict(y[:-self.n_preds])

        return {
            "mean_error": mad_mean_error(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "normal_error": nrmsd_error_norm(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "residual_error": residual_analysis(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "emc_error": rmse_train(y[-self.n_preds:], y_pred[-self.n_preds:])
        }

    def restore(self, save_path):
        pass

    def save(self, save_path):
        pass
