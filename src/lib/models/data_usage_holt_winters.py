#
#
#   Data usage Holt-Winters
#
#

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .model import Model
from .. import trackers
from ..utils.temporal_series_utils import estimated_window, initial_trend, initial_seasonal_components, \
    rmse_holt_winters
from ..utils.temporal_series_utils import mad_mean_error, nrmsd_error_norm, residual_analysis, rmse_train


class DataUsageHoltWinters(Model):

    def __init__(self, input_shape, **hyperparameters):
        super().__init__(input_shape)

        self.min_historic = hyperparameters.get("min_historic", 5)
        self.flag_optimize_hiperparams = hyperparameters.get("flag_optimize_hiperparams", True)
        self.n_preds = hyperparameters.get("n_preds", 3)
        self.slen = hyperparameters.get("slen", 1)
        self.alpha = hyperparameters.get("alpha", 0.716)
        self.beta = hyperparameters.get("beta", 0.029)
        self.gamma = hyperparameters.get("gamma", 0.993)

    def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        history = trackers.History()
        history.log_metrics({})
        return history.metrics

    def predict(self, X, batch_size=32, verbose=0):
        """
        Predict the next values of a series by applying the Triple Exponential Smoothing (a.k.a. Holt-Winters)

        """
        series = list(X)

        if self.flag_optimize_hiperparams:
            slen = estimated_window(series, 2)

            if (slen != -1) and (len(series) > int(self.min_historic)):
                initial_values = np.array([0.3, 0.1, 0.1])
                boundaries = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

                parameters = fmin_l_bfgs_b(rmse_holt_winters,
                                           x0=initial_values,
                                           args=(series[:], 'additive', self.slen),
                                           bounds=boundaries,
                                           approx_grad=True)

                self.alpha, self.beta, self.gamma = parameters[0]
                self.slen = slen

        cycle_days_past = self.input_shape[0]

        # Holt Winters model
        result = []
        smooth = series[0]
        trend = initial_trend(series, self.slen)
        seasonals = initial_seasonal_components(series, self.slen)

        result.append(series[0])

        for i in range(len(series) + self.n_preds - 1):

            # We are forecasting
            if i >= (len(series) - 1):
                m = i - len(series) + 1
                prediction = (smooth + m * trend) + seasonals[i % self.slen]

                if prediction < 0.0:
                    result.append(0.0)
                else:
                    result.append(prediction)

            # To know how much days user can continue with his normal consumption after his billing_end_date:
            else:
                last_smooth = smooth
                smooth = self.alpha * (series[i] - seasonals[i % self.slen]) + (1 - self.alpha) * (smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = (self.gamma * (series[i] - smooth) +
                                            (1 - self.gamma) * seasonals[i % self.slen])
                result.append(smooth + trend + seasonals[i % self.slen])

        cumsum = np.cumsum(series[-cycle_days_past:] + result[-self.n_preds:]).tolist()
        future_predictions = cumsum[-self.n_preds:]

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
            residual_error = 1.0
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
